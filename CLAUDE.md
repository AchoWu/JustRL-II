# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two layers live side-by-side:

- **`justrl2/`** — the recipe. All JustRL2-specific config, launch, data, model prep and eval scripts. Every knob lives in `justrl2/configs/*.env` and is overridable from the shell.
- **`miles/`, `miles_plugins/`, `train.py`, `train_async.py`, `tests/`, `tools/`** — a modified copy of the [Miles](https://github.com/radixark/miles) RL framework (SGLang rollout + Megatron-LM training + Ray). The recipe drives the framework, not the other way around. When touching framework files, remember they originate from Miles and are redistributed under Apache-2.0; keep changes minimal and idempotent where possible.
- **`third_party/patches/`** — patches for stock NVIDIA Megatron-LM / SGLang. Only used when *not* running on the `radixark/miles:dev` community image (which has them pre-applied).

`docs/method.md`, `docs/reproduce.md`, `docs/data.md` are the authoritative writeup — read them before proposing changes to the recipe.

## Environment

The training path requires the community image (Megatron-LM, SGLang, TransformerEngine, Ray, mbridge preinstalled). `justrl2/train.sh` expects `<repo>/Megatron-LM` and `<repo>/sglang` to exist (symlinked in the Dockerfile). Without those, training will not run; only pure-Python analysis / unit tests will.

```bash
docker build -t justrl2 . && docker run --gpus all --ipc=host --network=host -it justrl2
```

Pick the base tag by **host driver** — it decides the CUDA runtime:

| tag | CUDA | driver | platforms |
| --- | --- | --- | --- |
| `radixark/miles:dev` (default) | 13.0.3 | ≥ 580 | amd64 + arm64 |
| `radixark/miles:dev-cu12` | 12.9.2 | ≥ 525 | amd64 only |

`dev-cu12` is Miles' own `--variant cu12-x86` build (`lmsysorg/sglang:v0.5.19-cu129`, `ENABLE_CUDA_13=0`, `miles-wheels@cu129-x86_64`). Same Megatron-LM `miles-main` + `sglang-miles` as `dev`, with flash-attn (FA2 + FA3), TE 2.17 and apex prebuilt, so **no recipe or config change is needed** — use `1node-8gpu-32k.env`, not the `-baremetal` variant:

```bash
docker build --build-arg MILES_IMAGE=radixark/miles:dev-cu12 -t justrl2 .
```

Both tags rebuild daily with `MEGATRON_COMMIT` empty upstream (= branch HEAD), so pin a dated tag (`dev-cu12-202609130149`) for a run you intend to resume. On the CUDA 12 variant the P2P weight-transfer path is unavailable (base-image Mooncake, no structured object store); colocate mode never reaches it.

`justrl2/setup/bare_metal_cu129.sh` + `configs/1node-8gpu-32k-baremetal.env` are the **last resort** for hosts where docker itself is unavailable — `dev-cu12` supersedes them. Their three overrides (`ATTENTION_BACKEND=fused`, `NVTE_FUSED_ATTN=1`, `NO_GRAD_ACC_FUSION=1`) work around packages that are missing only on the hand-rolled stack; on the image they needlessly disable working fast paths.

`SKIP_PIP_INSTALL=1` is set inside the image so `justrl2/setup/setup.sh` only applies the Megatron patch (idempotent). `APPLY_MEGATRON_PATCH=1 bash justrl2/setup/setup.sh` is needed only when building on stock NVIDIA Megatron-LM.

Note: several files under `miles/` carry local compatibility shims for **Megatron/SGLang API drift** (vocab-padding helper moved to `megatron.training.vocab_utils`, `--enable-gloo-process-groups` → `--use-gloo-process-groups`, layer-spec signature filtering, sglang long-form parallelism flags, piecewise-cuda-graph knobs). These are **not** bare-metal workarounds — both images track branch HEAD, so the shims are required there too. Do not "revert" them when switching images.

## Common commands

Data + weights (one-time, need 1 GPU for the Megatron conversion):
```bash
bash   justrl2/prepare_model.sh    # openbmb/JustRL-II-base-model -> ./models (HF + torch_dist)
python justrl2/prepare_data.py     # -> ./datasets
```

Train (run on every node; Ray head is `RANK=0`):
```bash
bash justrl2/train.sh justrl2/configs/minicpm5-2b-math-128k.env
# override any knob from the shell:
GAE_LAMBDA_K=0.4 CRITIC_VALUE_BIAS_INIT=0.5 bash justrl2/train.sh justrl2/configs/minicpm5-2b-math-128k.env
# extra args after the config are forwarded to train.py
```

Debug pipeline on a single node (untested plumbing, not a small-scale recipe):
```bash
bash justrl2/train.sh justrl2/configs/debug-1node-8gpu.env
```

Resume: re-run the same command. If `NUM_ROLLOUT` changed on the first resume, add `OVERRIDE_OPT_PARAM_SCHEDULER=1`. A half-present actor/critic checkpoint pair is a hard error.

Offline eval of an HF export:
```bash
python justrl2/eval.py --model runs/<EXP_TAG>/hf/iter_0000299 \
    --data datasets/aime-2024.jsonl --data datasets/aime-2025.jsonl --data datasets/aime-2026.jsonl \
    --n 16 --temperature 1.0 --top-p 0.95 --max-tokens 126976
```

Recipe-only unit tests (no GPU / Megatron / SGLang needed):
```bash
python examples/value_head_demo.py
python -m pytest tests/test_gae_lambda_k.py tests/test_critic_value_bias_init.py tests/test_chunked_gae.py
```

Run a single pytest test:
```bash
python -m pytest tests/test_gae_lambda_k.py::<test_name> -vv
```

Tests are separated by folder: `tests/fast` (unit), `tests/e2e` (megatron/sglang/etc., need the stack), `tests/ci` (CI infra). `pyproject.toml` sets `testpaths = ["./tests"]` and skips `external/examples/docs/scripts/tools/tutorials`.

Style: black (line length 119), isort (black profile, `known_first_party = ["miles", "miles_plugins"]`), ruff (E/F/B/UP, `E402`/`E501` ignored, line length 320).

## Topology invariants (train.sh enforces these)

- PPO requires **actor world size == critic world size** — the actor↔critic NCCL groups are built rank-pairwise. `train.sh` defaults to `WORLD_SIZE/2` nodes each and refuses to start otherwise.
- Global batch (`ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`) must be divisible by actor DP (`nodes * gpus / TP / CP`).
- Reference run: 16 × 8 H100, 8 actor + 8 critic nodes, SGLang engines colocated on all GPUs, `TP=1 / CP=4`, GBS 480 (60 prompts × 8 samples).
- `SAVE_RETAIN_INTERVAL` must be a multiple of `SAVE_INTERVAL` (Megatron asserts).
- `MEGATRON_MODEL_PATH` must point at the parent `torch_dist` directory, **not** an `iter_xxx` subdirectory — otherwise Megatron silently starts from random weights.

## The three recipe knobs that matter (do not change casually)

| knob                     | value | why                                                                                          |
| ------------------------ | ----- | -------------------------------------------------------------------------------------------- |
| `GAE_LAMBDA_K`           | 0.5   | `λ_i = k^(1/L_i)`: constant terminal-credit fraction `k` at the first token regardless of length. Requires `GAMMA=1` (asserted). |
| `CRITIC_VALUE_BIAS_INIT` | 0.52  | Value head starts at the mean reward; removes the ~25-step warmup transient. Weight is zero-init. |
| `NUM_CRITIC_ONLY_STEPS`  | 30    | Critic converges before the first policy update. `actor.py` gates the policy update on `rollout_id >= num_critic_only_steps`. |

## Critical implementation details (easy to break silently)

- **Critic value-head re-init after finetune-style load.** `_rezero_critic_value_head` in `miles/backends/megatron_utils/checkpoint.py` runs after every finetune-style load: zeros the `[1, H]` head weight, re-fills the bias with `--critic-value-bias-init`, and calls `optimizer.reload_model_params()` so the fp32 master copy matches. Without this, the head inherits polluted values from the overlapping `[vocab, H]` LM head region and the run is silently broken. Startup log to grep for: `[critic-value-head] re-zeroed ... after policy-ckpt load (bias_init=0.52, master params resynced)`. Skipped on resume (`finetune=False`) so the trained head is kept.
- **Head-less critic.** The critic replaces `output_layer` with `LinearForLastLayer(hidden, 1)` (`miles/backends/megatron_utils/model_provider.py`) — no LM head, no token-prediction loss.
- **Length-adaptive λ is fp32-only.** At 128k, `λ = 1 − O(1e-5)` — bf16 rounds it to 1.0. `get_advantages_and_returns_batch` in `miles/utils/ppo_utils.py` computes λ in fp32 and only casts at the GAE scan.
- **Decoupled advantage vs value target.** Advantage uses per-sample `λ_i`; value target (`returns`) is the `λ=1` suffix-reward sum. The critic regresses an unbiased return while the actor uses the length-adaptive discount.
- **Overlong penalty excluded from critic target** when `CRITIC_EXCLUDE_OLP=1` (`--critic-exclude-overlong-penalty`). `critic_rewards` in `miles/ray/rollout.py` strips the DAPO soft overlong penalty so the value head models "will this be correct" rather than a correctness/length mixture. Actor advantages still see the penalty.
- **Save actor even during critic-only warmup.** `train.py:save` always saves the actor: resume infers `start_rollout_id` from the actor ckpt, so skipping the save during warmup would cause `start_rollout_id` to snap back to 0 while the critic keeps its trained state — misaligned resume.
- **Recompute granularity.** `--recompute-granularity full` at 128k. Switching to `selective` OOMs on single 127k samples.

## Startup checks worth grepping for

- `[critic-value-head] ... after-load ... w_absmax=0.189` then `re-zeroed [...] (bias_init=0.52, master params resynced)` — post-load re-init ran.
- First rollout sample responses are coherent text (not garbage — that indicates `MEGATRON_MODEL_PATH` pointed at an `iter_xxx` subdir).
- `critic exclude shaping: critic returns computed from rewards without overlong penalty`.

## DSpark speculative decoding

Only available in an SGLang build carrying the DSpark scheduler (not in the community image). Leave `DSPARK_DRAFT_MODEL_PATH` empty. It affects throughput only, not the recipe.

## What is generated vs source-controlled

`.gitignore` excludes `models/`, `datasets/`, `runs/`, `tensorboard/`, `swanlog/`, `*.pt`, `*.safetensors`. Do not commit any of these.
