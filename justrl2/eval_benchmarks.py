#!/usr/bin/env python3
"""Offline multi-benchmark evaluation of a JustRL2 checkpoint (the HF exports under
<SAVE_DIR>/hf/iter_XXXXXXX, or the base model).

    # all 12 benchmarks, 32 samples each
    python justrl2/eval_benchmarks.py --model runs/<EXP_TAG>/hf/iter_0000499

    # a subset, fewer samples, 16k-trained checkpoint
    python justrl2/eval_benchmarks.py --model runs/<EXP_TAG>/hf/iter_0000499 \
        --datasets aime-2024,aime-2025,aime-2026,gsm8k --n 8 --preset 16k

Metric: **pass@1 estimated from n independent samples** — for each problem the
fraction of its n samples that are correct, averaged over problems. That is the
unbiased estimator of single-sample accuracy, with a much smaller variance than one
sample per problem (the usual "avg@n" / "mean@n"). `pass@n` (solved at least once)
and `maj@n` (majority vote over extracted answers) are reported alongside as
diagnostics; the headline number is pass@1.

Sampling and prompting are aligned to training, which is the whole point of this
script rather than an ad-hoc harness:

| setting        | value          | where it comes from                              |
|----------------|----------------|--------------------------------------------------|
| prompt         | user turn only | `train.sh --input-key prompt --apply-chat-template`, no system message |
| answer format  | boxed suffix   | baked into every training `query`; `prepare_eval_data.py` re-appends it |
| temperature    | 1.0            | `ROLLOUT_TEMPERATURE` / `EVAL_TEMPERATURE`       |
| top_p          | 0.95           | `EVAL_TOP_P` (training *rollout* uses 1.0; the released AIME numbers use 0.95) |
| max_new_tokens | 30720 (32k)    | `ROLLOUT_MAX_RESPONSE_LEN` of the chosen preset  |
| grader         | `math` reward  | `--rm-type math` -> `grade_answer_union`         |

`--preset` must match the config the checkpoint was *trained* with, because the
response-length cap is not a free parameter here: a 16k-trained policy evaluated
with a 128k budget is being asked for lengths it never produced, and a 128k-trained
one evaluated at 16k gets truncated mid-derivation. `--max-tokens` overrides it.

Needs SGLang importable (the community image) and one GPU (more with --tp).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Run as `python justrl2/eval_benchmarks.py`: sys.path[0] is `justrl2/`, so neither the
# repo root nor the sglang checkout is importable. Add both -- and put `sglang/python`
# FIRST: `<repo>/sglang` is a source checkout whose package lives one level down at
# `sglang/python/sglang`, so a repo root ahead of it makes the bare `<repo>/sglang/`
# directory win as an implicit namespace package and shadow the real install
# (`import sglang` then has `__file__ is None` and no `srt`). train.sh sets
# PYTHONPATH=.:Megatron-LM:${SGLANG_PATH}/python for the same reason.
_PREPEND = [
    Path(__file__).resolve().parent.parent / "sglang" / "python",
    Path(__file__).resolve().parent.parent,
]
sys.path[:0] = [str(p) for p in _PREPEND if p.is_dir()]

# Default order: hardest first, so a run killed early still has the headline sets.
DEFAULT_DATASETS = [
    "aime-2024",
    "aime-2025",
    "aime-2026",
    "olympiadbench",
    "gsm8k",
    "minerva-math",
    "svamp",
    "asdiv",
    "mawps",
    "tabmwp",
    "mmlu-stem",
    "sat-math",
]

# (max_new_tokens, context_len) per training config. Keys match justrl2/configs/.
PRESETS = {
    "16k": (14336, 16384),
    "32k": (30720, 32768),
    "128k": (126976, 131072),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval -- a sane error bar for a 30-problem set."""
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom


# ----------------------------------------------------------------- grading
# Grading is the second-largest cost after generation and it is pure CPU: a *failing*
# grade falls through `grade_answer_verl` (sympy, ~20 ms in-process) into the
# math-verify subprocess fallback (~160 ms), so a 3018-problem set at n=32 would spend
# hours of wall-clock in the grader alone. Two fixes, both verified against serial
# grading on mixed real rows:
#   * dedup by (problem, boxed answer) -- n samples collapse onto a few answers;
#   * a process pool -- measured 2.6x on 8 processes. Threads do NOT help (1.1x):
#     `math_verify_fallback` picks "the first worker whose lock is free" without
#     holding a lock across the check, so concurrent threads all pile onto worker[0].
# The task must therefore be a module-level function over picklable data, not the
# closure that reads `args`.

_GradeTask = tuple  # (answer | None, label, kind, choices_json, mc_value_credit)


def _grade_task(task: _GradeTask) -> bool:
    """One grade, in a worker process.

    miles' `math` reward (`--rm-type math` -> `grade_answer_union`), plus
    letter<->option-text credit for multiple choice. A multiple-choice item has one
    right answer but two faithful spellings of it: the letter ("C") and the option
    text ("reduce the carrying capacity ..."). The label is the letter and the grader
    compares strings, so a response that boxes the correct *text* would score 0 on a
    question it answered right. Accept either; --no-mc-value-credit is letter-only.
    """
    from miles.rollout.rm_hub.math_utils import grade_answer_union

    answer, label, kind, choices_json, mc_credit = task
    if answer is None:
        return False
    if grade_answer_union(f"\\boxed{{{answer}}}", label):
        return True
    if not mc_credit or kind != "mc":
        return False
    choices = json.loads(choices_json) if choices_json else {}
    gold_text = choices.get(label.upper())
    if not gold_text:
        return False
    given = answer.strip().strip("$").strip()
    # "(C)" / "C." spellings; "\text{C}" is already handled by the grader above.
    if given.strip("(). ").upper() == label.upper():
        return True
    return given == gold_text.strip() or grade_answer_union(f"\\boxed{{{given}}}", gold_text)


def _make_task(answer: str | None, row: dict, mc_credit: bool) -> _GradeTask:
    meta = row.get("metadata") or {}
    choices = meta.get("choices")
    return (
        answer,
        str(row["label"]).strip(),
        meta.get("kind"),
        json.dumps(choices, sort_keys=True) if choices else None,
        mc_credit,
    )


def _detect_preset(model_path: str) -> str | None:
    """Guess the training length config from the checkpoint path.

    Every config sets an EXP_TAG carrying its length (`justrl2_minicpm5_2b_math16k_1node`,
    `..._math128k`), and runs land under `runs/<EXP_TAG>/hf/iter_XXXXXXX`, so the path
    usually says which preset the policy was trained with. Only a hint: --preset wins.
    """
    s = str(model_path).replace("\\", "/").lower()
    for key in ("128k", "32k", "16k"):  # longest first: "128k" also ends in "8k"
        if key in s:
            return key
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF checkpoint dir (hf/iter_XXXXXXX or base)")
    ap.add_argument("--data-dir", default="datasets/eval", help="from justrl2/prepare_eval_data.py")
    ap.add_argument(
        "--datasets",
        default="all",
        help=f"comma-separated subset, or 'all' (default). available: {','.join(DEFAULT_DATASETS)}",
    )
    ap.add_argument("--data", action="append", default=[], help="extra jsonl path with prompt/label; repeatable")
    ap.add_argument("--n", type=int, default=32, help="independent samples per problem (default 32)")
    ap.add_argument("--limit", type=int, default=None, help="first N problems per dataset (smoke tests)")
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="training length config; default = guessed from the checkpoint path, else 32k",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=None, help="override the preset response cap")
    ap.add_argument("--context-len", type=int, default=None, help="override the preset context length")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--max-running-requests", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=8192, help="requests per generate() call (memory bound)")
    ap.add_argument("--out", default=None, help="write per-sample records to this jsonl")
    ap.add_argument("--summary", default=None, help="write the summary table to this json")
    ap.add_argument(
        "--grader-workers",
        type=int,
        default=min(16, (os.cpu_count() or 8)),
        help="processes for the math grader (1 = serial, in-process)",
    )
    ap.add_argument(
        "--no-mc-value-credit",
        action="store_true",
        help="for multiple-choice sets, require the boxed answer to be the letter (default also accepts the option text)",
    )
    args = ap.parse_args()

    preset_source = "explicit"
    if args.preset is None:
        detected = _detect_preset(args.model)
        args.preset, preset_source = (detected, "from path") if detected else ("32k", "default")

    max_tokens = args.max_tokens if args.max_tokens is not None else PRESETS[args.preset][0]
    context_len = args.context_len if args.context_len is not None else PRESETS[args.preset][1]
    if max_tokens >= context_len:
        ap.error(f"--max-tokens {max_tokens} must be < --context-len {context_len} (prompt needs room too)")

    # ---- resolve the files before loading the engine (a typo should fail in 1s) ----
    names = DEFAULT_DATASETS if args.datasets == "all" else [s.strip() for s in args.datasets.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    files: list[tuple[str, Path]] = []
    missing = []
    for name in names:
        path = data_dir / f"{name}.jsonl"
        (files.append((name, path)) if path.is_file() else missing.append(str(path)))
    for extra in args.data:
        p = Path(extra)
        (files.append((p.stem, p)) if p.is_file() else missing.append(extra))
    if missing:
        ap.error("missing data file(s):\n  " + "\n  ".join(missing) + "\n\nrun: python justrl2/prepare_eval_data.py")
    if not files:
        ap.error("no datasets selected")

    datasets = []
    for name, path in files:
        rows = load_jsonl(path)
        if args.limit:
            rows = rows[: args.limit]
        if rows:
            datasets.append((name, rows))
    n_req = sum(len(rows) for _, rows in datasets) * args.n

    # Each grader process gets its own math_verify_fallback worker pool, built at
    # import time from MILES_MATH_VERIFY_WORKERS. One subprocess per grader process is
    # the right shape -- a grader process issues one fallback call at a time, and the
    # default of 2 would double the subprocess count for no gain.
    os.environ.setdefault("MILES_MATH_VERIFY_WORKERS", "1")

    import sglang as sgl
    from transformers import AutoTokenizer

    from miles.rollout.rm_hub.math_utils import extract_answer

    print(
        f"model={args.model}\n"
        f"preset={args.preset} ({preset_source})  max_new_tokens={max_tokens}  context_len={context_len}\n"
        f"n={args.n}  temperature={args.temperature}  top_p={args.top_p}  top_k={args.top_k}  seed={args.seed}\n"
        f"{len(datasets)} dataset(s), {sum(len(r) for _, r in datasets)} problems, {n_req} requests\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        tp_size=args.tp,
        dp_size=args.dp,
        context_length=context_len,
        mem_fraction_static=args.mem_fraction,
        random_seed=args.seed,
        # train.sh feeds the context length to max_position_embeddings the same way;
        # without it a model whose config caps at 4k silently truncates long prompts.
        json_model_override_args=json.dumps({"max_position_embeddings": context_len}),
        log_level="warning",
    )
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    engine = sgl.Engine(**engine_kwargs)

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": max_tokens,
    }
    if args.top_k > 0:
        sampling["top_k"] = args.top_k

    # Grading runs in worker processes (see _grade_task). Spawned before the first
    # dataset and reused, since each worker pays a one-off sympy/math_verify import.
    pool = cf.ProcessPoolExecutor(max_workers=args.grader_workers) if args.grader_workers > 1 else None

    def grade_many(answers: list[str | None], rows_for: list[dict]) -> list[bool]:
        """Grade a batch: dedup by (problem, boxed answer), then fan out to processes.

        Dedup is sound because grading depends on the response *only* through
        `extract_answer`: `grade_answer_union` takes the last \\boxed{} payload, and a
        None extraction grades False for every label (`grade_answer_verl` returns
        False on `given_answer is None`; the math-verify fallback is fail-closed on a
        None pred). So two responses with the same boxed payload always grade the
        same, and an unboxed response is always 0 -- verified empirically too.

        Keyed on the task tuple, not `id(row)`: two rows of the same dataset can share
        a label and choice set, and then they genuinely are the same grading question.
        """
        tasks = [_make_task(a, rows_for[i], not args.no_mc_value_credit) for i, a in enumerate(answers)]
        uniq: dict[_GradeTask, bool] = {}
        todo = list(dict.fromkeys(tasks))

        if pool is not None and len(todo) > 1:
            results = pool.map(_grade_task, todo, chunksize=max(1, len(todo) // (args.grader_workers * 4) or 1))
        else:
            results = map(_grade_task, todo)
        for task, ok in zip(todo, results, strict=True):
            uniq[task] = ok
        return [uniq[t] for t in tasks]

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    summary: list[dict] = []

    for name, rows in datasets:
        t0 = time.time()
        # Index i of the flat request list maps to problem i // n, sample i % n.
        prompts = []
        for row in rows:
            p = row["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        flat = [p for p in prompts for _ in range(args.n)]

        outputs: list[dict] = []
        for start in range(0, len(flat), args.chunk_size):
            outputs.extend(engine.generate(flat[start : start + args.chunk_size], sampling))

        per_problem: list[list[bool]] = [[] for _ in rows]
        answers: list[list[str]] = [[] for _ in rows]
        lengths, truncated = [], []
        # Extract first, grade as one batch: dedup + threads turn the ~160 ms cost of
        # a failing grade from a per-sample serial tax into a per-distinct-answer one.
        extracted: list[str | None] = []
        rows_for: list[dict] = []
        for i, o in enumerate(outputs):
            q = i // args.n
            ans = extract_answer(o["text"])
            extracted.append(ans.strip() if ans is not None else None)
            rows_for.append(rows[q])
            lengths.append(o["meta_info"]["completion_tokens"])
            truncated.append(o["meta_info"].get("finish_reason", {}).get("type") == "length")
        graded = grade_many(extracted, rows_for)

        for i, o in enumerate(outputs):
            q, s = divmod(i, args.n)
            ok = graded[i]
            per_problem[q].append(ok)
            answers[q].append(extracted[i] or "")
            if out_f:
                out_f.write(
                    json.dumps(
                        {
                            "dataset": name,
                            "idx": q,
                            "sample": s,
                            "correct": ok,
                            "label": rows[q]["label"],
                            "answer": extracted[i],
                            "tokens": lengths[i],
                            "truncated": truncated[i],
                            "response": o["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # pass@1: per-problem correct fraction, averaged over problems. Equal weight
        # per problem (not per sample) -- identical when every problem has n samples,
        # but robust if a request is dropped.
        per_problem_rate = [statistics.mean(v) for v in per_problem if v]
        pass1 = statistics.mean(per_problem_rate)
        pass_n = statistics.mean(any(v) for v in per_problem if v)
        solved_all = statistics.mean(all(v) for v in per_problem if v)
        # maj@n: plurality vote over extracted answers, graded once per problem. The
        # vote is over raw answer *strings*, so "0.5" and "\frac{1}{2}" are distinct
        # candidates even though the grader would equate them -- the usual convention,
        # and the reason maj@n is a diagnostic here rather than the headline metric.
        maj_answers: list[str | None] = []
        for votes in answers:
            cand = [a for a in votes if a]
            maj_answers.append(collections.Counter(cand).most_common(1)[0][0] if cand else None)
        maj_n = statistics.mean(grade_many(maj_answers, list(rows)))
        no_box = statistics.mean(not a for votes in answers for a in votes)

        rec = dict(
            dataset=name,
            problems=len(rows),
            n=args.n,
            pass_at_1=pass1,
            pass_at_1_ci95=_wilson_halfwidth(pass1, len(rows)),
            pass_at_n=pass_n,
            maj_at_n=maj_n,
            solved_all=solved_all,
            mean_len=statistics.mean(lengths),
            p95_len=sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            truncated=statistics.mean(truncated),
            no_boxed_answer=no_box,
            seconds=time.time() - t0,
        )
        summary.append(rec)
        print(
            f"{name:<14} {len(rows):>5}x{args.n:<3} "
            f"pass@1={pass1 * 100:6.2f}+-{rec['pass_at_1_ci95'] * 100:4.2f}  "
            f"pass@{args.n}={pass_n * 100:6.2f}  maj@{args.n}={maj_n * 100:6.2f}  "
            f"len={rec['mean_len']:6.0f}/p95={rec['p95_len']:<6d} "
            f"trunc={rec['truncated'] * 100:5.2f}%  nobox={no_box * 100:5.2f}%  "
            f"[{rec['seconds'] / 60:.1f}m]",
            flush=True,
        )

    if out_f:
        out_f.close()
    if pool is not None:
        pool.shutdown()
    engine.shutdown()

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 92)
    print(
        f"{'dataset':<16}{'problems':>9}{'pass@1':>10}{'+-95%':>8}{f'pass@{args.n}':>10}{f'maj@{args.n}':>10}{'trunc%':>9}"
    )
    print("-" * 92)
    for r in summary:
        print(
            f"{r['dataset']:<16}{r['problems']:>9}{r['pass_at_1'] * 100:>10.2f}"
            f"{r['pass_at_1_ci95'] * 100:>8.2f}{r['pass_at_n'] * 100:>10.2f}"
            f"{r['maj_at_n'] * 100:>10.2f}{r['truncated'] * 100:>9.2f}"
        )
    print("-" * 92)
    # Unweighted mean over datasets: the conventional way these suites are reported,
    # so a 32-problem set is not drowned out by a 3018-problem one.
    print(
        f"{'macro average':<16}{sum(r['problems'] for r in summary):>9}"
        f"{statistics.mean(r['pass_at_1'] for r in summary) * 100:>10.2f}"
    )
    print("=" * 92)

    if args.summary:
        meta = dict(
            model=args.model,
            n=args.n,
            preset=args.preset,
            max_new_tokens=max_tokens,
            context_len=context_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            metric="pass@1 = mean over problems of (correct samples / n)",
        )
        Path(args.summary).write_text(
            json.dumps({"config": meta, "results": summary}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nsummary -> {args.summary}")


if __name__ == "__main__":
    main()
