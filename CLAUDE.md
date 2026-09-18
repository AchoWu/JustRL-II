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

Single-node wrappers around `train.sh` (they add pre-flight checks, tmpfs weight
staging and post-run assertions; `train.sh` remains the real launcher):
```bash
bash run_test.sh              # 3-rollout smoke: GBS 16, 2k responses, pipeline only
bash run_train.sh --dry-run   # pre-flight only, prints the resolved command
bash run_train.sh --probe     # 3 rollouts under a separate EXP_TAG to measure step time/VRAM
bash run_train.sh --16k       # 16k variant (default; see below)
bash run_train.sh --32k       # 32k, the recipe's 1-node form
```
`run_train.sh` stages `models/` into `/dev/shm/llms` before loading (HF weight load
went 8m42s -> 0.8s). `SHM_WEIGHTS=0` opts out, `SHM_DIR` moves it. It reserves room
for Ray's plasma store, which also lives in `/dev/shm`.

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

`1node-8gpu-16k{,-baremetal}.env` is **not a reproduction of the recipe**, even though
all five knobs above (plus `CRITIC_EXCLUDE_OLP`, `GAMMA`) are inherited untouched.
`λ_i = k^(1/L_i)` takes response length as its input, so capping at 14336 instead of
30720 shifts the whole λ distribution — same `k`, shorter `L`, smaller λ, stronger
discounting. Its curves are not comparable to the 32k or 128k ones. It exists because
32k over-subscribes the KV cache on this node (see below). Four values move together
there: response len, context len, overlong buffer (20% of the response cap), and
concurrency — context len matters as much as response len, since `train.sh` feeds it
to `max_position_embeddings`, `sglang-context-length` and `sglang-max-prefill-tokens`.

## Critical implementation details (easy to break silently)

- **Critic value-head re-init after finetune-style load.** `_rezero_critic_value_head` in `miles/backends/megatron_utils/checkpoint.py` runs after every finetune-style load: zeros the `[1, H]` head weight, re-fills the bias with `--critic-value-bias-init`, and calls `optimizer.reload_model_params()` so the fp32 master copy matches. Without this, the head inherits polluted values from the overlapping `[vocab, H]` LM head region and the run is silently broken. Startup log to grep for: `[critic-value-head] re-zeroed ... after policy-ckpt load (bias_init=0.52, master params resynced)`. Skipped on resume (`finetune=False`) so the trained head is kept.
- **Head-less critic.** The critic replaces `output_layer` with `LinearForLastLayer(hidden, 1)` (`miles/backends/megatron_utils/model_provider.py`) — no LM head, no token-prediction loss.
- **Length-adaptive λ is fp32-only.** At 128k, `λ = 1 − O(1e-5)` — bf16 rounds it to 1.0. `get_advantages_and_returns_batch` in `miles/utils/ppo_utils.py` computes λ in fp32 and only casts at the GAE scan.
- **Decoupled advantage vs value target.** Advantage uses per-sample `λ_i`; value target (`returns`) is the `λ=1` suffix-reward sum. The critic regresses an unbiased return while the actor uses the length-adaptive discount.
- **Overlong penalty excluded from critic target** when `CRITIC_EXCLUDE_OLP=1` (`--critic-exclude-overlong-penalty`). `critic_rewards` in `miles/ray/rollout.py` strips the DAPO soft overlong penalty so the value head models "will this be correct" rather than a correctness/length mixture. Actor advantages still see the penalty.
- **Save actor even during critic-only warmup.** `train.py:save` always saves the actor: resume infers `start_rollout_id` from the actor ckpt, so skipping the save during warmup would cause `start_rollout_id` to snap back to 0 while the critic keeps its trained state — misaligned resume.
- **Recompute granularity.** `--recompute-granularity full` at 128k. Switching to `selective` OOMs on single 127k samples.
- **`ReloadableProcessGroup` must never return `None` from a collective.** Since torch 2.8 `ProcessGroupNCCL::collectiveCoalesced` ends `return asyncOp ? work : nullptr`, so a *synchronous* coalesced collective hands back `nullptr` → Python `None`. `PyProcessGroup`'s `WORK_OVERRIDE` macro does not check: it feeds that into `make_intrusive<PyWorkHolder>(o)`, and because `Work`'s holder is `PYBIND11_DECLARE_HOLDER_TYPE(..., true)`, `None` casts to an *empty* `intrusive_ptr` instead of being rejected. `PyWorkHolder::wait` then does `work_->wait(timeout)` with no null check → SIGSEGV on every rank at once, at the critic's first gradient reduction. `_fwd` in `miles/utils/reloadable_process_group.py` returns a synthetic `_CompletedWork` instead (`None` means the collective already finished synchronously, so `wait() == True` is correct). **Upgrading torch does not fix this**: the `o.is_none()` guard from pytorch#189817 is absent in 2.13 and present in 2.14, but `reduce_scatter_tensor_coalesced` is one of the sites that bypass `WORK_OVERRIDE` with a hand-written dual-name lookup, still unguarded in 2.14. Nor is 2.13 ahead of upstream — miles pins it transitively via `lmsysorg/sglang:v0.5.19`. Also resolve the `reduce_scatter_tensor` → `reduce_scatter_single` rename: `PyProcessGroup` looks up the `*_single*` spelling first, so both are defined and dispatched against whichever the inner PG implements.
- **Two co-loaded cuDNN copies segfault TE's FusedAttention.** `conda install -c nvidia cudnn` (needed for `cudnn.h` when building TE's torch extension) also drops a full runtime into `$CONDA_PREFIX/lib`. Both it and pip's copy carry SONAME `libcudnn.so.9`; whichever loads first with RTLD_GLOBAL wins symbol resolution, then dlopens its own sub-libraries by SONAME and gets the other version → `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` at the critic's first forward, 70 minutes into startup. Move the runtime libs **outside** `$CONDA_PREFIX`, keeping the header: a subdirectory of `lib/` is *not* enough (verified — `lib/_shadowed/` created 2609-09-15 was still mapped by the 09-16 run). This is separate from the ldconfig-cache copy handled in `bare_metal_cu129.sh`; both need plugging. `verify_env.py`'s "cuDNN 单一来源" check reads `/proc/self/maps` and is the only one of the four cuDNN checks that catches this — version, `CDLL`, and ldconfig all pass on a broken machine.
- **KV cache is over-subscribed at 32k on one node.** The pool is `SGLANG_MAX_TOTAL_TOKENS` *per engine* and each concurrent request reserves its full context: `32 x 30720 = 983k` against a 524k pool, 1.9x over. Early steps fit because samples are short; once response length grows (what RL does early on) SGLang starts logging `KV cache pool is full. Retract requests.` and thrashes — a retract discards generated KV and re-prefills, so the work is lost and rollout wedges (observed: 50 minutes at 0/64 samples). Concurrency must come down with the length; halving the length alone lands on `32 x 16384 = 524288`, exactly at the limit.


## OPEN BUG: the critic's value head never trains (bare-metal cu129 / H20)

**Status: unresolved. Do not start a long run until this is closed** — the recipe's
three core knobs all exist to shape a working critic, so with a dead one the whole
thing degenerates to something GRPO-ish.

Symptom, straight out of a checkpoint after 24 steps of training:

```
output_layer.weight: absmax=0 mean=0 nonzero=0/2048     <- never moved
output_layer.bias:   0.51953125                         <- still the 0.52 init
```

`V(s)` is therefore a constant and PPO's advantage collapses to plain reward.

The causal chain, each link measured (`[vh-diag]` logging in `model.py`,
`model_provider.py`, `checkpoint.py`):

```
hidden states entering output_layer are all zero  (absmax=0, nonzero=0/30146560)
  -> dV/dw = input_ = 0, so weight.main_grad == 0
  -> dV/db = 1 regardless, so bias.main_grad == 0.484  (this asymmetry is the tell)
  -> weight never moves; V(s) == bias for every state
```

The decoder's own output is already zero, while its graph is intact
(`requires_grad=True`, `grad_fn=MakeViewlessTensorBackward`) — so this is a real
zero, not a severed backward path.

### Ruled out (all measured, not argued)

| Hypothesis | How it died |
| --- | --- |
| Checkpoint is corrupt / missing layers | All 9 weight tensors fully non-zero, sane std (qkv 0.027, fc1 0.027, emb 0.024). 42 layers present — megatron packs them one key per module with the layer as a shard dim, so `decoder.layers.self_attention.linear_qkv.weight` is a single `(42, 2560, 2048)` entry. 180 keys is normal for that packing; `layers.0.*` matches nothing by design |
| tmpfs staging broke the copy | `/dev/shm` copy byte-identical to source: same size, file count, key count, values |
| `MEGATRON_MODEL_PATH` at the wrong level | Layout is correct (`latest_checkpointed_iteration.txt` + `release/`), parent dir as required |
| `allow_shape_mismatch` causing a silent partial load | `Megatron-LM/megatron/core/utils.py:997` only `logger.warning`s the leftover kwargs and forwards them verbatim. Noise, not a load change |
| FP8/FP4 quantization path | `fp8=None`, `fp4=None`; miles-main's NVFP4 QAT is not involved |
| Chunked-logits bypass skipping `output_layer` | Gate logs `use_chunked=False role='critic'` — `post_process` stays on |
| Sequence-parallel gather severing the gradient | `sequence_parallel=False` on this path |
| `value_loss_function`'s `torch.max(surr1, surr2)` | At step 0 `values == old_values` so `surr1 == surr2`; reproduced offline and `w.grad` is 1.045, not 0 |
| Warmup pinning lr at 0 | `lr_warmup_steps = 10*64 = 640` vs `lr_decay_steps = 500*64 = 32000`. And at rollout 1 `lr=5e-07` while grad is still 0 |
| `is_embedding_or_output_parameter` skewing param groups | Both megatron uses are gated on `decoupled_lr_enabled`; `decoupled_lr` is None |

### Three readings that look like evidence and are not

Each of these cost a round of investigation:

- **`critic-grad_norm` (0.79–0.89) does not mean the head has gradients.** It is
  `optimizer.get_grad_norm()`, a whole-model norm; a 2048-element head barely
  moves it. Note `value_loss=0.3962` against `grad_norm=0.7924` is exactly the
  factor of 2 that `d/dV (V-R)^2 = 2(V-R)` gives, so a gradient *w.r.t. V* does
  exist — it just never reaches the weight.
- **`main_absmax` in the `[critic-value-head]` lines is weight-OR-bias.**
  `_value_head_main_param_absmax` takes the max over both, so the familiar
  `0.5196` is the *bias*, and says nothing about the weight.
- **Comparing `id()` against `optimizer.param_groups` always reports 0.** The
  distributed optimizer stores fp32 *master* copies there, never the model's bf16
  params. A "head not in optimizer" reading from that is a false negative.

Also: coherent rollout samples prove nothing about the megatron-side forward —
rollout text comes from SGLang, which loads its own copy of the weights.

### Where it stands

Zeros originate inside the forward. `_attach_vh_input_probe` samples four points in
order — `embedding-out`, `layer0-out`, `final_layernorm`, `decoder-out` — for
**both** roles, so the first zero localizes the origin, and the actor/critic
comparison shows whether this is critic-specific or global. If the actor is also
zero its `log_probs` are degenerate too, which silently breaks TIS and the KL term.

```bash
grep "vh-diag" nohup.out | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "(critic|actor)/[a-z0-9_-]+ \[.*"
```

Untested suspect if all four come back non-zero: `--recompute-granularity full`,
the one remaining setting that changes forward behaviour and has not been varied.

## Startup checks worth grepping for

- `[critic-value-head] ... after-load ... w_absmax=0.189` then `re-zeroed [...] (bias_init=0.52, master params resynced)` — post-load re-init ran.
- First rollout sample responses are coherent text (not garbage — that indicates `MEGATRON_MODEL_PATH` pointed at an `iter_xxx` subdir).
- `critic exclude shaping: critic returns computed from rewards without overlong penalty`.
- `[vh-diag] ... decoder-out ...` with a non-zero `absmax` — see the open bug above.


## DSpark speculative decoding

Only available in an SGLang build carrying the DSpark scheduler (not in the community image). Leave `DSPARK_DRAFT_MODEL_PATH` empty. It affects throughput only, not the recipe.

## What is generated vs source-controlled

`.gitignore` excludes `models/`, `datasets/`, `runs/`, `tensorboard/`, `swanlog/`, `*.pt`, `*.safetensors`. Do not commit any of these.
