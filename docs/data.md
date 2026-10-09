# Data

## Training set — `openbmb/UltraData-RL-2609`, config `Math`

JustRL2 trains on the **Math** slice of
[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609):
**32,412** competition- and textbook-style problems, each with a single extractable
answer verified against `ground_truth`. `python justrl2/prepare_data.py` downloads it
and writes `<data-dir>/UltraData-RL-Math-2609.jsonl`.

The Hub schema is five fields:

| field | meaning |
|---|---|
| `uuid` | `Math_00001`-style id |
| `query` | the problem statement |
| `ground_truth` | the reference answer |
| `source` | provenance tag |
| `domain` | `Math` |

`prepare_data.py` adds `prompt` (= `query`) and `label` (= `ground_truth`) because that
is what miles reads (`--input-key prompt --label-key label`); the original columns are
kept as metadata. `prompt` goes through the model's chat template as a single user turn,
and responses are graded by miles' `math` reward (rule-based normalisation with
math-verify as a fallback).

### Why the difficulty distribution matters here

The dataset card documents that difficulty is calibrated **against the RL initialization
checkpoint** — the same `openbmb/JustRL-II-base-model` this recipe starts from: items it
already solves at pass rate 1 are dropped (no gradient), the learnable band is kept, and
pass-rate-0 items with a confirmed-valid label are kept and left to online dynamic
sampling. That is why `DYNAMIC_SAMPLING=1` is part of the recipe: it discards
zero-variance groups at rollout time, so the remaining budget goes to problems that still
produce a learning signal. Labels are never modified by that filtering.

If you swap in your own corpus, the property to preserve is that one: drop what the
starting checkpoint already solves every time, keep the mixed band.

## Evaluation — AIME 2024 / 2025 / 2026

The eval sets are public competition benchmarks and are **not** part of
UltraData-RL-2609, so `prepare_data.py` does not download them by default. Provide three
jsonl files with the same `prompt` / `label` fields:

```
<data-dir>/aime-2024.jsonl
<data-dir>/aime-2025.jsonl
<data-dir>/aime-2026.jsonl
```

and `train.sh` picks them up (`TEST_FILE="aime2024 … aime2025 … aime2026 …"`). Several
AIME sets are mirrored on the Hub; if the one you use is laid out as splits or configs of
a single repo, `prepare_data.py --eval-repo <repo> --eval-splits aime2024,aime2025,aime2026`
converts them for you. The reported numbers use 30 problems per year, 16 samples per
problem, T=1.0, top-p 0.95, 126976-token budget (`justrl2/eval.py`).

## The broader eval suite — 12 math/STEM benchmarks

`python justrl2/prepare_eval_data.py` downloads and converts twelve sets into
`datasets/eval/<name>.jsonl`, with the same `prompt` / `label` schema, for
`justrl2/eval_benchmarks.py`:

| set | problems | kind | source |
|---|---|---|---|
| `aime-2024` | 30 | math | `math-ai/aime24` (answer is in `solution` as `\boxed{...}`) |
| `aime-2025` | 30 | math | `math-ai/aime25` |
| `aime-2026` | 30 | math | `math-ai/aime26` |
| `olympiadbench` | 675 | math | Qwen2.5-Math mirror; `final_answer[0]`, `$` stripped |
| `gsm8k` | 1319 | math | Qwen2.5-Math mirror; label after `####` |
| `minerva-math` | 272 | math | Qwen2.5-Math mirror; `\boxed{}` out of `solution` (272/272 have one) |
| `svamp` | 1000 | math | body + question concatenated |
| `asdiv` | 2215 | math | body + question; the `(apples)` unit is stripped from the label |
| `mawps` | 2065 | math | `input` / `target` |
| `tabmwp` | 1000 | math | table prepended to the question; numeric labels canonicalised |
| `mmlu-stem` | 3018 | **mc** | Qwen2.5-Math mirror, 18 STEM subjects (4 options) |
| `sat-math` | 32 | **mc** | Qwen2.5-Math mirror (4 options) |

11,686 problems total. Nine come from the
[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) evaluation mirror — one uniform
jsonl schema, no `datasets` dependency, and the surface form these benchmarks are
conventionally reported in. The question/answer shaping follows that harness'
`parser.py` (`parse_question` / `parse_ground_truth`).

Minerva Math is 272 problems, and that is the whole set — not a truncation. It is the
OCW/MIT undergraduate STEM slice from the Minerva paper, and five independent mirrors
(`math-ai/minervamath`, `svc-huggingface/minerva-math`, `zwhe99/simplerl-minerva-math`,
`1231czx/minerva_math`, `nanoverl/minerva`) all carry exactly 272 rows over the same
problems. Any "1000-problem Minerva" is a different benchmark — most likely MATH-500
or the 1000-row TabMWP/SVAMP subsets, which are separate entries above.

Two things the conversion does beyond reformatting:

1. **It appends the training answer instruction.** Every `prompt` ends with
   `\nPlease reason step by step, and put your final answer within \boxed{}.` because
   every training `query` does (5988/5988 checked rows) and the grader needs a
   `\boxed{}` to extract anything. The raw problem stays in `question`, so
   `--no-answer-suffix` ablates it without re-downloading.
2. **Multiple-choice rows keep their options.** `metadata.kind = "mc"` and
   `metadata.choices` (letter -> option text) let the eval credit a response that
   boxes the correct option *text* rather than the letter — same answer, and the label
   is the letter. Everything else is `kind = "math"`.

`--datasets a,b,c` restricts the download; existing files are skipped unless
`--overwrite`.

## Using your own data

Any jsonl with `prompt` and `label` works:

```bash
export TRAIN_FILE="/path/a.jsonl /path/b.jsonl"     # space-separated, mixed together
export TEST_FILE="mytest /path/test.jsonl"           # name path [name path ...]
```

Labels must be gradable by the math verifier — a number, expression, or short answer.
