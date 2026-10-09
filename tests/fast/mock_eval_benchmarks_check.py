"""Drive eval_benchmarks.main() with a fake sglang engine + tokenizer.

Verifies: prompt assembly (chat template, single user turn, boxed suffix intact),
the i -> (problem, sample) mapping, pass@1 / pass@n / maj@n arithmetic, MC
letter-vs-text credit, truncation + no-box accounting, chunking, summary json,
and that parallel grading agrees exactly with serial grading.

    python tests/fast/mock_eval_benchmarks_check.py <dir-from-prepare_eval_data>

Needs no GPU and no sglang. The `__main__` guard and the plain `import
eval_benchmarks` (rather than a spec load under an ad-hoc name) are both required:
the grader's ProcessPoolExecutor uses spawn on Windows, so every worker re-imports
this module and must be able to import the module that `_grade_task` lives in.
"""

import collections
import json
import random
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BS = chr(92)  # backslash, kept out of string literals to dodge escape confusion
SUFFIX = "put your final answer within " + BS + "boxed{}."


def boxed(x):
    return "reasoning... " + BS + "boxed{" + str(x) + "}"


class FakeEngine:
    last = None

    def __init__(self, **kw):
        self.kw = kw
        self.calls = []
        FakeEngine.last = self

    def generate(self, prompts, sampling):
        self.calls.append(len(prompts))
        self.sampling = sampling
        out = []
        for p in prompts:
            gold = p.split("<<GOLD:")[1].split(">>")[0]
            mode = p.split("<<MODE:")[1].split(">>")[0]
            text = p.split("<<TEXT:")[1].split(">>")[0]
            r = random.random()
            if mode == "mc_text":
                body = boxed(text)  # always answers with the option TEXT, never the letter
            elif r < 0.5:
                body = boxed(gold)
            elif r < 0.7:
                body = boxed(99999)
            else:
                body = "no box at all"
            out.append(
                {
                    "text": body,
                    "meta_info": {
                        "completion_tokens": random.randint(50, 500),
                        "finish_reason": {"type": "length" if r > 0.95 else "stop"},
                    },
                }
            )
        return out

    def shutdown(self):
        pass


ROWS = {}


class FakeTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        assert len(msgs) == 1 and msgs[0]["role"] == "user", msgs
        content = msgs[0]["content"]
        assert content.endswith(SUFFIX), repr(content[-70:])
        row = ROWS[content]
        meta = row.get("metadata") or {}
        mode, text = "plain", ""
        if meta.get("kind") == "mc" and meta.get("choices"):
            mode, text = "mc_text", meta["choices"][row["label"]]
        return (
            "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
            "<<GOLD:" + row["label"] + ">><<MODE:" + mode + ">><<TEXT:" + text + ">>"
        )


class FakeAT:
    @staticmethod
    def from_pretrained(*a, **k):
        return FakeTok()


# Stub sglang/transformers at import time (not inside the guard) so the spawned
# grader workers re-importing this module do not try to load the real packages.
fake_sgl = types.ModuleType("sglang")
fake_sgl.Engine = FakeEngine
sys.modules.setdefault("sglang", fake_sgl)
fake_tf = types.ModuleType("transformers")
fake_tf.AutoTokenizer = FakeAT
sys.modules.setdefault("transformers", fake_tf)

sys.path.insert(0, str(REPO / "justrl2"))
import eval_benchmarks as mod  # noqa: E402  (must follow the stubs above)


def run(data_dir: Path, workers: int, out: str, summary: str):
    random.seed(0)
    sys.argv = [
        "eval_benchmarks.py",
        "--model",
        "fake",
        "--data-dir",
        str(data_dir),
        "--datasets",
        "aime-2024,gsm8k,mmlu-stem,sat-math,tabmwp",
        "--n",
        str(N),
        "--limit",
        str(LIMIT),
        "--preset",
        "16k",
        "--chunk-size",
        "64",
        "--grader-workers",
        str(workers),
        "--summary",
        summary,
        "--out",
        out,
    ]
    mod.main()
    return (
        [json.loads(line) for line in open(out, encoding="utf-8")],
        json.loads(Path(summary).read_text(encoding="utf-8")),
    )


N, LIMIT, NDS = 8, 20, 5

if __name__ == "__main__":
    DATA = Path(sys.argv[1])
    for f in DATA.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            ROWS[r["prompt"]] = r

    recs, s = run(DATA, 8, "/tmp/_rec.jsonl", "/tmp/_sum.json")

    print("\n--- harness assertions ---")
    e = FakeEngine.last
    assert e.kw["context_length"] == 16384, e.kw["context_length"]
    assert json.loads(e.kw["json_model_override_args"])["max_position_embeddings"] == 16384
    assert e.sampling == {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 14336}, e.sampling
    assert max(e.calls) <= 64, e.calls
    print("engine kwargs + sampling + chunking OK; calls:", len(e.calls), "max chunk:", max(e.calls))

    assert len(recs) == NDS * LIMIT * N, (len(recs), NDS * LIMIT * N)
    counts = set(collections.Counter((r["dataset"], r["idx"]) for r in recs).values())
    assert counts == {N}, counts
    samples = set(collections.Counter((r["dataset"], r["sample"]) for r in recs).values())
    assert samples == {LIMIT}, samples
    print("record count/mapping OK:", len(recs), "records,", counts, "samples per problem")

    res = {r["dataset"]: r for r in s["results"]}
    for name, r in res.items():
        byq = collections.defaultdict(list)
        for rec in recs:
            if rec["dataset"] == name:
                byq[rec["idx"]].append(rec["correct"])
        recomputed = sum(sum(v) / len(v) for v in byq.values()) / len(byq)
        assert abs(recomputed - r["pass_at_1"]) < 1e-12, (name, recomputed, r["pass_at_1"])
    print("pass@1 arithmetic matches per-sample records for all", len(res), "datasets")

    for name in ("mmlu-stem", "sat-math"):
        assert res[name]["pass_at_1"] == 1.0, (name, res[name]["pass_at_1"])
        assert res[name]["maj_at_n"] == 1.0, (name, res[name]["maj_at_n"])
    print("MC letter<->text credit OK (mmlu-stem/sat-math all 1.000)")

    for name, r in res.items():
        assert r["pass_at_1"] <= r["pass_at_n"] + 1e-12, name
        assert r["solved_all"] <= r["pass_at_1"] + 1e-12, name
    for name in ("aime-2024", "gsm8k", "tabmwp"):
        assert 0.2 < res[name]["pass_at_1"] < 0.8, (name, res[name]["pass_at_1"])
        assert res[name]["no_boxed_answer"] > 0.1, name
    print("monotonicity pass@1 <= pass@n and no-box accounting OK")

    # Serial grading must reproduce the parallel numbers exactly.
    print("\n--- re-running with --grader-workers 1 ---")
    recs1, s1 = run(DATA, 1, "/tmp/_rec1.jsonl", "/tmp/_sum1.json")
    res1 = {r["dataset"]: r for r in s1["results"]}
    for name in res:
        for k in ("pass_at_1", "pass_at_n", "maj_at_n", "solved_all", "no_boxed_answer"):
            assert res[name][k] == res1[name][k], (name, k, res[name][k], res1[name][k])
    assert [r["correct"] for r in recs] == [r["correct"] for r in recs1]
    print("parallel grading == serial grading, exactly")

    print("\nALL MOCK CHECKS PASSED")
