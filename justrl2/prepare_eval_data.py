#!/usr/bin/env python3
"""Download the 12 math/STEM benchmarks and convert them to the `prompt` / `label`
jsonl that `justrl2/eval_benchmarks.py` (and `train.sh`'s TEST_FILE) read.

    python justrl2/prepare_eval_data.py                      # all 12 -> datasets/eval/
    python justrl2/prepare_eval_data.py --datasets gsm8k,aime-2025
    python justrl2/prepare_eval_data.py --no-answer-suffix    # ablate the boxed instruction

Two conventions are reproduced here, and both matter for the numbers to mean anything:

1. **The prompt is the training prompt.** Every row of the training corpus
   (`openbmb/UltraData-RL-2609`, config Math) ends with the *same* literal line

       \\nPlease reason step by step, and put your final answer within \\boxed{}.

   (verified: 5988/5988 rows of `Math_part-1-of-4.jsonl`). It is part of `query`,
   i.e. part of the user turn, *not* a system prompt — `JustRL-II-base-model`'s
   `chat_template.jinja` emits a `<|im_start|>system` block only when the caller
   passes one, and `train.sh` passes `--input-key prompt --apply-chat-template`
   with no system message. So the suffix is appended here, into `prompt`, and the
   eval applies the chat template as a single user turn with nothing else.
   `question` keeps the raw problem text so the suffix can be changed or dropped
   without re-downloading.

2. **The question/answer surface form follows Qwen2.5-Math's evaluation harness**
   (`evaluation/parser.py::parse_question` / `parse_ground_truth`), since that is
   the form these benchmarks are conventionally reported in: ASDiv/SVAMP
   concatenate body + question, TabMWP prepends the table, and the two
   multiple-choice sets (MMLU-STEM, SAT-Math) render options as
   `Answer Choices: (A) ... (B) ...` with a *letter* label.

Multiple-choice rows carry `metadata.kind = "mc"` plus `metadata.choices` (the
letter -> option-text map) so the grader can also accept a boxed option *value*;
everything else is `kind = "math"` and graded by miles' `math` reward verbatim.

Sources: the Qwen2.5-Math mirror on GitHub for the 9 conventional sets (one uniform
schema, no `datasets` dependency) and `math-ai/aime{24,25,26}` for AIME.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# The instruction line that every training prompt ends with. Appended to `question`
# after a single newline, exactly as in the training corpus.
ANSWER_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."

QWEN_BASE = "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/{name}/test.jsonl"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

LETTERS = "ABCD"  # MMLU-STEM and SAT-Math are both 4-option


# --------------------------------------------------------------------------- io


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "justrl2-prepare-eval-data"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _fetch_jsonl(url: str) -> list[dict]:
    rows = []
    for line in _fetch(url).decode("utf-8").split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _fetch_parquet(url: str) -> list[dict]:
    import pandas as pd

    return pd.read_parquet(url).to_dict("records")


def _qwen(name: str) -> list[dict]:
    return _fetch_jsonl(QWEN_BASE.format(name=name))


# ------------------------------------------------------------------ formatting


def _fmt_num(x) -> str:
    """Canonicalise a numeric label: 43.0 -> '43', 34.350 -> '34.35'."""
    f = float(x)
    if f == int(f):
        return str(int(f))
    return f"{f:.10f}".rstrip("0").rstrip(".")


def _boxed(text: str) -> str | None:
    """Last \\boxed{...} payload, brace-matched."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:  # \boxed 5  (no braces)
        return text[i + len("\\boxed") :].strip().split("$")[0].strip() or None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k].strip()
    return None


def _choices_block(choices: dict[str, str]) -> str:
    return "Answer Choices: " + " ".join(f"({k}) {str(v).strip()}" for k, v in choices.items())


# ------------------------------------------------------------------ converters
# Each converter yields (question, label, extra) where `extra` lands in metadata.
# `label` is always a string, gradable by miles' `math` reward.


def _conv_aime24(r: dict) -> tuple[str, str, dict]:
    # math-ai/aime24 stores the answer as `solution = "\boxed{204}"`.
    return r["problem"].strip(), (_boxed(r["solution"]) or r["solution"]).strip(), {}


def _conv_aime(r: dict) -> tuple[str, str, dict]:
    return r["problem"].strip(), str(r["answer"]).strip(), {}


def _conv_gsm8k(r: dict) -> tuple[str, str, dict]:
    return r["question"].strip(), r["answer"].split("####")[-1].strip().replace(",", ""), {}


def _conv_minerva(r: dict) -> tuple[str, str, dict]:
    return r["problem"].strip(), (_boxed(r["solution"]) or "").strip(), {}


def _conv_olympiad(r: dict) -> tuple[str, str, dict]:
    q = r["question"].strip()
    if r.get("context"):
        q = f"{r['context'].strip()}\n{q}"
    return q, str(r["final_answer"][0]).strip().strip("$").strip(), {"subfield": r.get("subfield")}


def _conv_svamp(r: dict) -> tuple[str, str, dict]:
    body = r["Body"].strip()
    if not body.endswith("."):
        body += "."
    return f"{body} {r['Question'].strip()}", _fmt_num(r["Answer"]), {}


def _conv_asdiv(r: dict) -> tuple[str, str, dict]:
    # answers look like "9 (apples)" -- the unit in parentheses is not part of it.
    label = re.sub(r"\(.*?\)", "", str(r["answer"])).strip()
    return f"{r['body'].strip()} {r['question'].strip()}", label, {}


def _conv_mawps(r: dict) -> tuple[str, str, dict]:
    return r["input"].strip(), _fmt_num(r["target"]), {}


def _conv_tabmwp(r: dict) -> tuple[str, str, dict]:
    title = f'regarding "{r["table_title"]}" ' if r.get("table_title") else ""
    q = f"Read the following table {title}and answer a question:\n{r['table']}\n{r['question'].strip()}"
    if r.get("choices"):
        q += f" Please select from the following options: {list(r['choices'])}"
    ans = str(r["answer"]).strip()
    if r.get("ans_type") in ("integer_number", "decimal_number"):
        if "/" in ans:
            num, den = ans.split("/")[:2]
            ans = _fmt_num(float(num) / float(den))
        elif "%" in ans:
            ans = _fmt_num(float(ans.split("%")[0]) / 100)
        else:
            ans = _fmt_num(ans.replace(",", ""))
    return q, ans, {}


def _conv_mmlu_stem(r: dict) -> tuple[str, str, dict]:
    choices = {LETTERS[i]: str(c).strip() for i, c in enumerate(r["choices"])}
    q = f"{r['question'].strip()}\n{_choices_block(choices)}"
    return q, LETTERS[int(r["answer"])], {"kind": "mc", "choices": choices, "subject": r.get("type")}


def _conv_sat_math(r: dict) -> tuple[str, str, dict]:
    # `options` is one flat string. Most rows use "A) $x$ B) $y$ ...", but four
    # (ids 17/26/27/29) use "A. $x$ B. $y$ ..." -- accept either delimiter. The
    # leading `(?:^|\s)` keeps a stray "... point A. Then" inside option *text*
    # from splitting, since only a standalone letter at a token boundary matches.
    opts = r["options"].strip()
    parts = re.split(r"(?:^|\s)([A-D])[).]\s+", opts)
    choices = {}
    for k in range(1, len(parts) - 1, 2):
        choices[parts[k]] = parts[k + 1].strip()
    q = f"{r['question'].strip()}\n{_choices_block(choices) if choices else opts}"
    return q, str(r["Answer"]).strip(), {"kind": "mc", "choices": choices}


# -------------------------------------------------------------------- registry

DATASETS: dict[str, dict] = {
    "aime-2024": dict(
        loader=lambda: _fetch_parquet(HF_RESOLVE.format(repo="math-ai/aime24", path="test-00000-of-00001.parquet")),
        conv=_conv_aime24,
        n=30,
    ),
    "aime-2025": dict(
        loader=lambda: _fetch_jsonl(HF_RESOLVE.format(repo="math-ai/aime25", path="test.jsonl")),
        conv=_conv_aime,
        n=30,
    ),
    "aime-2026": dict(
        loader=lambda: _fetch_jsonl(HF_RESOLVE.format(repo="math-ai/aime26", path="aime2026.jsonl")),
        conv=_conv_aime,
        n=30,
    ),
    "olympiadbench": dict(loader=lambda: _qwen("olympiadbench"), conv=_conv_olympiad, n=675),
    "gsm8k": dict(loader=lambda: _qwen("gsm8k"), conv=_conv_gsm8k, n=1319),
    "minerva-math": dict(loader=lambda: _qwen("minerva_math"), conv=_conv_minerva, n=272),
    "svamp": dict(loader=lambda: _qwen("svamp"), conv=_conv_svamp, n=1000),
    "asdiv": dict(loader=lambda: _qwen("asdiv"), conv=_conv_asdiv, n=2215),
    "mawps": dict(loader=lambda: _qwen("mawps"), conv=_conv_mawps, n=2065),
    "tabmwp": dict(loader=lambda: _qwen("tabmwp"), conv=_conv_tabmwp, n=1000),
    "mmlu-stem": dict(loader=lambda: _qwen("mmlu_stem"), conv=_conv_mmlu_stem, n=3018),
    "sat-math": dict(loader=lambda: _qwen("sat_math"), conv=_conv_sat_math, n=32),
}

ALL = list(DATASETS)


def build_rows(name: str, raw: list[dict], answer_suffix: str | None) -> list[dict]:
    conv = DATASETS[name]["conv"]
    rows, dropped = [], 0
    for i, r in enumerate(raw):
        try:
            question, label, extra = conv(dict(r))
        except Exception as e:  # a malformed upstream row must not kill the file
            dropped += 1
            print(f"  ! {name}[{i}]: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if not question.strip() or not str(label).strip():
            dropped += 1
            continue
        prompt = f"{question}\n{answer_suffix}" if answer_suffix else question
        meta = {"dataset": name, "idx": i, "kind": extra.pop("kind", "math")}
        meta.update({k: v for k, v in extra.items() if v is not None})
        rows.append({"prompt": prompt, "label": str(label), "question": question, "metadata": meta})
    if dropped:
        print(f"  ({dropped} rows dropped: empty question/label or bad schema)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="datasets/eval", help="output directory (default: datasets/eval)")
    ap.add_argument("--datasets", default="all", help=f"comma-separated subset of: {','.join(ALL)}")
    ap.add_argument(
        "--no-answer-suffix",
        action="store_true",
        help="do NOT append the training boxed instruction (ablation; the grader needs \\boxed{})",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    names = ALL if args.datasets == "all" else [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(unknown)}\navailable: {', '.join(ALL)}")

    suffix = None if args.no_answer_suffix else ANSWER_SUFFIX
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total, failed = 0, []
    for name in names:
        path = out_dir / f"{name}.jsonl"
        if path.exists() and not args.overwrite:
            n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            print(f"{name:<14} {n:>5} rows  (exists, skipped; --overwrite to refresh)")
            total += n
            continue
        print(f"{name:<14} downloading ...", flush=True)
        try:
            raw = DATASETS[name]["loader"]()
        except Exception as e:
            print(f"{name:<14} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(name)
            continue
        rows = build_rows(name, raw, suffix)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        expected = DATASETS[name]["n"]
        flag = "" if len(rows) == expected else f"  (expected {expected})"
        print(f"{name:<14} {len(rows):>5} rows -> {path}{flag}")
        total += len(rows)

    print(f"\n{total} problems across {len(names) - len(failed)} datasets in {out_dir.resolve()}")
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("\nnext:\n  python justrl2/eval_benchmarks.py --model runs/<EXP_TAG>/hf/iter_0000499 --n 32")


if __name__ == "__main__":
    main()
