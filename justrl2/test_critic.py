#!/usr/bin/env python3
"""Use a JustRL2 critic straight from its Megatron dist checkpoint — no HF export.

    # 1. 只看 value head，秒级，不加载 backbone（最常用）
    python justrl2/test_critic.py --critic runs/<TAG>_critic/iter_0000499

    # 2. 真跑前向，算 V(s)，并检查它能不能区分对错
    python justrl2/test_critic.py --critic runs/<TAG>_critic/iter_0000499 \
        --base models/JustRL-II-base-model \
        --samples samples.jsonl --device cuda

Nothing is written to disk. The dist checkpoint is read with
`torch.distributed.checkpoint` directly (`no_dist=True`, single process), the
megatron->HF name mapping is applied in memory, and the result is loaded into a plain
`AutoModel` backbone plus an `nn.Linear(H, 1)` head. So this needs no Megatron, no
SGLang, no Ray, and no second copy of the weights. `--inspect` does not even load the
backbone.

The value head has to be intercepted rather than run through the shared converter:
the critic's `output_layer` is `LinearForLastLayer(hidden, 1)`, and the per-model
converters map `output_layer.weight` to `lm_head.weight` (wrong — the critic has no
LM head) and raise `ValueError: Unknown parameter name` on `output_layer.bias`.

The value semantics match training exactly, and the alignment is the easy thing to
get wrong:

    V(s_t) = head(h[prompt_len + t - 1])

i.e. the value of response token `t` is read off the hidden state of the token
*before* it (`loss.py:get_responses` slices `logits[start - 1 : end - 1]`). So
`V(s_0)` — the value of the prompt alone, before a single response token exists — is
the output at the last prompt position. That is the number to look at: with
`CRITIC_EXCLUDE_OLP=1` the critic regresses the overlong-penalty-free return, so
`V(s_0)` is the critic's estimate of P(this prompt gets answered correctly).

`--samples` takes jsonl with `prompt` and `response`, plus either `correct` (bool) or
`label` (graded with the same `math` grader training used). `justrl2/eval.py --out`
writes `response`/`correct` but not `prompt`, so pass `--join-data` with the original
AIME/UltraData jsonl to join them back by row index.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dist_cp

_REPO_ROOT = Path(__file__).resolve().parent.parent

# `<repo>/sglang` is a checkout of the sglang source tree; the importable package sits
# one level down at `sglang/python/sglang`. So putting the repo root at sys.path[0]
# makes the bare `<repo>/sglang/` directory win as an implicit namespace package and
# shadow the real install: `import sglang` then yields a module with `__file__ = None`
# and no `srt` submodule, and every `sglang.srt.*` import under miles dies. train.sh
# sidesteps this with PYTHONPATH=.:Megatron-LM:${SGLANG_PATH}/python — do the same,
# putting the checkout's python/ dir ahead of the repo root so the real package is
# found first either way.
#
# `miles` and `tools` come from the repo root; it is normally `pip install -e .` (see
# bare_metal_cu129.sh), so that entry is belt-and-braces for a plain checkout.
REPO_ROOT = Path("/group/40092/howu/JustRL-II")

_PREPEND = [
    REPO_ROOT / "sglang" / "python",
    REPO_ROOT,
]

sys.path[:0] = [str(p) for p in _PREPEND if p.is_dir()]

VALUE_HEAD_WEIGHT = "output_layer.weight"
VALUE_HEAD_BIAS = "output_layer.bias"

# rms of the critic's final hidden state, measured from the [vh-diag] forward probe
# (critic/decoder-out std=3.8-4.0 on the 16k run). Only used to turn ||w||_2 into an
# estimate of V's spread in --inspect, which needs no forward pass; --samples measures
# the spread directly and does not use this.
HIDDEN_RMS_ESTIMATE = 4.0


def load_dist_state_dict(input_dir: Path) -> tuple[dict, object | None]:
    """Read a Megatron dist checkpoint into a plain state dict, on CPU, single process.

    The tensors live in the `__*.distcp` shards and are indexed by the `.metadata`
    dotfile; that pair is the whole checkpoint as far as `torch.distributed.checkpoint`
    is concerned. `common.pt` is a *separate* file carrying the pickled megatron
    Namespace, and megatron only writes it in some configurations — the training
    checkpoints on this recipe do not have one. So it is optional here: returns
    `None` when absent, and `megatron_args_from_hf_config` rebuilds what the name
    mapping needs from the HF config instead.
    """
    from tools.convert_torch_dist_to_hf import EmptyStateDictLoadPlanner, WrappedStorageReader

    megatron_args = None
    common = input_dir / "common.pt"
    if common.is_file():
        # Importing the tools module above installs its pickle stub for megatron
        # classes, which unpickling this Namespace needs.
        megatron_args = torch.load(common, weights_only=False)["args"]

    state_dict: dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(input_dir)),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict, megatron_args


def megatron_args_from_hf_config(base_hf_dir: Path, state_dict: dict):
    """Rebuild the handful of megatron args the name mapping needs, from the HF config.

    Only six fields are ever read: `num_layers` and `num_experts` (to unpack megatron's
    packed-layer dimension) and `hidden_size` / `num_attention_heads` /
    `num_query_groups` / `kv_channels` (to split the fused QKV). Every one of them is
    in the HF config the model was converted from, so a missing `common.pt` is not a
    blocker. `vocab_size` is deliberately absent: it only feeds `remove_padding`, and
    the critic has no vocab-shaped tensor to unpad.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_hf_dir, trust_remote_code=True)
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    return SimpleNamespace(
        num_layers=cfg.num_hidden_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_query_groups=getattr(cfg, "num_key_value_heads", None) or heads,
        kv_channels=getattr(cfg, "head_dim", None) or hidden // heads,
        vocab_size=cfg.vocab_size,
        num_experts=getattr(cfg, "num_experts", None),
        q_lora_rank=None,
        sglang_enable_ep_moe=False,
    )


def inspect_value_head(state_dict: dict, megatron_args) -> dict:
    """Report the value head without touching the backbone.

    This is the check worth running on its own: an all-zero weight with the bias still
    at its init is the signature of the CPU-backup bug (see the resolved section in
    CLAUDE.md), and no training metric reports it — `critic-grad_norm` is a
    whole-model norm that 2048 elements barely move.
    """
    if VALUE_HEAD_WEIGHT not in state_dict:
        raise SystemExit(f"no {VALUE_HEAD_WEIGHT} in the checkpoint — not a critic checkpoint")
    w = state_dict[VALUE_HEAD_WEIGHT]
    if w.dim() != 2 or w.shape[0] != 1:
        raise SystemExit(
            f"{VALUE_HEAD_WEIGHT} has shape {tuple(w.shape)}, expected [1, hidden]. "
            "This is a policy checkpoint (its LM head is [vocab, hidden]), not a critic."
        )
    b = state_dict.get(VALUE_HEAD_BIAS)
    wf = w.float()
    bias_init = getattr(megatron_args, "critic_value_bias_init", None)

    # V = w·h + b, so how much V can vary across states is set by ||w||_2, not by the
    # per-element rms. With h's components at roughly hidden-independent scale,
    # std(w·h) ~= ||w||_2 * rms(h); the measured rms(h) at the critic's final
    # layernorm is ~4 on this recipe ([vh-diag] critic/decoder-out std=3.8-4.0), which
    # turns ||w||_2 into a usable estimate of V's spread before running any forward.
    w_l2 = float(wf.norm())
    info = {
        "hidden_size": w.shape[1],
        "nonzero": int(wf.ne(0).sum()),
        "numel": wf.numel(),
        "absmax": float(wf.abs().max()),
        "rms": float(wf.pow(2).mean().sqrt()),
        "w_l2": w_l2,
        "est_v_spread": w_l2 * HIDDEN_RMS_ESTIMATE,
        "bias": float(b.float().item()) if b is not None else None,
        "bias_init": bias_init,
        "gae_lambda_k": getattr(megatron_args, "gae_lambda_k", None),
        "critic_exclude_overlong_penalty": getattr(megatron_args, "critic_exclude_overlong_penalty", None),
        "critic_lr": getattr(megatron_args, "critic_lr", None),
    }

    print(f"  value head      [1, {info['hidden_size']}]")
    print(
        f"    weight        nonzero={info['nonzero']}/{info['numel']} "
        f"absmax={info['absmax']:.6g} rms={info['rms']:.6g} L2={w_l2:.4g}"
    )
    print(f"    bias          {info['bias']!r}  (init was {bias_init!r})")
    if megatron_args is None:
        # These four come only from common.pt, which megatron did not write for this
        # checkpoint. Say so once instead of printing four bare Nones that look like
        # the recipe knobs were unset.
        print("  recipe knobs    (unavailable: no common.pt in this checkpoint)")
    else:
        print(f"  gae_lambda_k    {info['gae_lambda_k']}")
        print(f"  exclude OLP     {info['critic_exclude_overlong_penalty']}")
        print(f"  critic_lr       {info['critic_lr']}")
    print(f"  est. V spread   ~+-{info['est_v_spread']:.4f} around the {info['bias']:.4f} prior")
    print(f"                  (= ||w||_2 x rms(h), rms(h)~{HIDDEN_RMS_ESTIMATE:g}; --samples measures it for real)")

    if info["nonzero"] == 0:
        print("\n  !! weight is ALL ZERO — the critic never trained. V(s) is a constant.")
        print("     See 'Resolved: the critic's value head never trained' in CLAUDE.md.")
    elif info["est_v_spread"] < 0.05:
        # The thing actually worth warning about. Deliberately NOT "the bias did not
        # move": the bias is stored in bf16, one ulp at 0.52 is 2^-8 = 0.0039, and at
        # critic_lr 5e-6 it takes ~390 same-sign steps to shift one tick — so a bias
        # printing its init exactly is the normal case on a healthy run, and warning
        # on it would cry wolf every time. What matters is whether V can separate
        # states at all, which is ||w||_2's job.
        print(
            f"\n  ?? V(s) varies by only ~{info['est_v_spread']:.3f} across states, "
            f"against a {info['bias']:.3f} prior."
        )
        print("     The head is close to constant, so PPO's advantage is close to whitened reward.")
        print("     Run with --samples to measure the real spread and the correct-vs-wrong AUC.")
    return info


def convert_backbone(state_dict: dict, megatron_args, model_name: str) -> dict:
    """Map the critic's backbone params to HF names, holding back the value head.

    The head has to be skipped before `convert_to_hf` sees it: the per-model
    converters would turn `output_layer.weight` into `lm_head.weight` and raise on
    `output_layer.bias`. Everything else goes through the shared dispatch untouched.
    """
    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from tools.convert_torch_dist_to_hf import get_named_params

    hf_tensors: dict = {}
    for name, param in get_named_params(megatron_args, state_dict):
        if name.endswith((VALUE_HEAD_WEIGHT, VALUE_HEAD_BIAS)):
            continue
        for hf_name, hf_param in convert_to_hf(megatron_args, model_name, name, param):
            hf_tensors[hf_name] = hf_param
    return hf_tensors


def build_critic(state_dict: dict, megatron_args, base_hf_dir: Path, model_name: str | None, device: str, dtype):
    """Assemble the critic in memory: HF backbone + the scalar head. Writes nothing."""
    from transformers import AutoConfig, AutoModel

    if model_name is None:
        model_name = type(AutoConfig.from_pretrained(base_hf_dir, trust_remote_code=True)).__name__.lower()
    if megatron_args is None:
        megatron_args = megatron_args_from_hf_config(base_hf_dir, state_dict)
    # Re-check the shape here and not only in inspect_value_head: pointed at an actor
    # checkpoint, output_layer.weight is [vocab, H], and silently taking its first row
    # as a value head would produce numbers that look like values and are not.
    head_weight = state_dict[VALUE_HEAD_WEIGHT]
    if head_weight.dim() != 2 or head_weight.shape[0] != 1:
        raise SystemExit(
            f"{VALUE_HEAD_WEIGHT} has shape {tuple(head_weight.shape)}, expected [1, hidden] — "
            "this is a policy checkpoint, not a critic."
        )
    head_bias = state_dict[VALUE_HEAD_BIAS]
    hf_tensors = convert_backbone(state_dict, megatron_args, model_name)

    config = AutoConfig.from_pretrained(base_hf_dir, trust_remote_code=True)
    # AutoModel is the bare backbone (LlamaModel, not LlamaForCausalLM), which is
    # exactly what the critic is: the LM head was replaced before training started.
    backbone = AutoModel.from_config(config, trust_remote_code=True)

    # convert_to_hf emits HF *CausalLM names ("model.embed_tokens.weight"); the bare
    # backbone drops that prefix.
    backbone_sd = {k[len("model.") :]: v for k, v in hf_tensors.items() if k.startswith("model.")}
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    # rotary inv_freq and friends are buffers the config rebuilds; real missing weights
    # would mean the name mapping is wrong and V(s) would be garbage.
    missing = [k for k in missing if not k.endswith(("inv_freq", "rotary_emb.inv_freq"))]
    if missing or unexpected:
        raise SystemExit(f"backbone load mismatch — missing={missing[:8]} unexpected={unexpected[:8]}")

    head = torch.nn.Linear(head_weight.shape[1], 1, bias=True)
    head.weight.data.copy_(head_weight)
    head.bias.data.copy_(head_bias)

    backbone = backbone.to(device=device, dtype=dtype).eval()
    head = head.to(device=device, dtype=dtype).eval()
    return backbone, head, config


@torch.no_grad()
def compute_values(backbone, head, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Per-response-token values for one sample, aligned exactly as training aligns them.

    Returns `[R]` where element `t` is `V(s_t)`, the value of the state *before*
    response token `t` — i.e. the head applied to hidden state `prompt_len + t - 1`.
    Element 0 is therefore the value of the prompt alone.
    """
    hidden = backbone(input_ids=input_ids).last_hidden_state  # [1, T, H]
    values = head(hidden).squeeze(-1).squeeze(0)  # [T]
    total_len = input_ids.shape[1]
    # get_responses: logits[start - 1 : end - 1] with end = total_len, start = end - R.
    return values[prompt_len - 1 : total_len - 1].float()


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Rank-based AUC: P(V of a correct sample > V of an incorrect one). 0.5 = useless."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_samples(path: Path, join_data: Path | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if join_data is not None:
        # eval.py --out records carry idx/response/correct but not the prompt.
        src = [json.loads(line) for line in join_data.open(encoding="utf-8") if line.strip()]
        for r in rows:
            if "prompt" not in r:
                if "idx" not in r:
                    raise SystemExit("--join-data needs an 'idx' field in the samples file (eval.py --out has it)")
                r["prompt"] = src[r["idx"]]["prompt"]
                r.setdefault("label", src[r["idx"]].get("label"))
    for r in rows:
        if "prompt" not in r or "response" not in r:
            raise SystemExit("each sample needs 'prompt' and 'response' (see --join-data for eval.py output)")
    return rows


def grade(row: dict) -> bool | None:
    if "correct" in row:
        return bool(row["correct"])
    if row.get("label") is None:
        return None
    from miles.rollout.rm_hub.math_utils import grade_answer_union

    return bool(grade_answer_union(row["response"], str(row["label"])))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. Used for V(s_0) vs per-prompt accuracy, where n is small."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        out = [0.0] * n
        i = 0
        while i < n:  # average ranks within ties, or the coefficient is biased
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _report_per_prompt(records: list[dict]) -> None:
    """Group by V(s_0) to recover per-prompt behaviour.

    V(s_0) is a function of the prompt alone, so every sample of the same prompt shares
    it exactly — which means it doubles as a prompt id here, and the number of distinct
    values is the real sample size for any claim about V(s_0). With 8 samples per
    prompt a 64-row table is 8 prompts, and a sample-level AUC on V(s_0) is dominated
    by which prompts happened to be easy, not by whether V ranks them.
    """
    groups: dict[float, list[dict]] = {}
    for r in records:
        if r["correct"] is not None:
            groups.setdefault(round(r["v0"], 4), []).append(r)
    if len(groups) < 2:
        return

    print(
        f"\n  per-prompt (V(s_0) is a function of the prompt, so it groups them): "
        f"{len(groups)} distinct prompts"
    )
    print(f"    {'V(s_0)':>8} {'n':>3} {'acc':>6}")
    v0s, accs = [], []
    for v0 in sorted(groups):
        rows = groups[v0]
        acc = sum(1 for r in rows if r["correct"]) / len(rows)
        v0s.append(v0)
        accs.append(acc)
        print(f"    {v0:>8.4f} {len(rows):>3} {acc:>6.3f}")

    rho = spearman(v0s, accs)
    if rho is not None:
        print(f"    Spearman(V(s_0), per-prompt acc) = {rho:+.3f} over {len(groups)} prompts")
        if abs(rho) < 0.3:
            print("    -> V(s_0) does not rank prompt difficulty. The head's prompt-level")
            print("       signal is noise; only the response-conditioned values carry any.")


def run_samples(args, state_dict, megatron_args) -> None:
    from transformers import AutoTokenizer

    rows = load_samples(Path(args.samples), Path(args.join_data) if args.join_data else None)
    if args.limit:
        rows = rows[: args.limit]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"\nbuilding critic in memory (device={args.device}, dtype={args.dtype}) ...")
    backbone, head, _ = build_critic(state_dict, megatron_args, Path(args.base), args.model_name, args.device, dtype)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"scoring {len(rows)} samples\n")
    print(f"  {'#':>4} {'V(s_0)':>9} {'V_mean':>9} {'V_last':>9} {'len':>6}  correct")
    v0_correct: list[float] = []
    v0_wrong: list[float] = []
    records = []
    for i, row in enumerate(rows):
        p = row["prompt"]
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        response_ids = tok(row["response"], add_special_tokens=False).input_ids
        if not response_ids:
            continue
        input_ids = torch.tensor([prompt_ids + response_ids], device=args.device)

        values = compute_values(backbone, head, input_ids, len(prompt_ids))
        ok = grade(row)
        v0 = float(values[0])
        if ok is True:
            v0_correct.append(v0)
        elif ok is False:
            v0_wrong.append(v0)
        records.append(
            {
                "idx": i,
                "v0": v0,
                "v_mean": float(values.mean()),
                "v_last": float(values[-1]),
                "n_response": len(response_ids),
                "correct": ok,
            }
        )
        print(
            f"  {i:>4} {v0:>9.4f} {float(values.mean()):>9.4f} {float(values[-1]):>9.4f} "
            f"{len(response_ids):>6}  {ok}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nper-sample values -> {args.out}")

    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)
    all_v0 = [r["v0"] for r in records]
    if not all_v0:
        print("  没有可评分的样本")
        return
    spread = max(all_v0) - min(all_v0)
    print(f"  V(s_0)          mean={statistics.mean(all_v0):.4f} min={min(all_v0):.4f} max={max(all_v0):.4f}")
    print(f"  spread          {spread:.4f}")
    if spread < 1e-3:
        # A head whose weight never grew produces V == bias for every input. That is
        # not a crash, it is PPO silently degrading to plain reward (adv = R - const).
        print("    !! V(s) is effectively constant across prompts — the head carries no signal.")
        print("       PPO's advantage would collapse to whitened reward. Check --inspect output.")

    if v0_correct and v0_wrong:
        # Report AUC for all three, because V(s_0) alone is the least informative and
        # reading only it has already produced one wrong conclusion. V(s_0) depends on
        # the PROMPT only, so with n samples per prompt its effective sample size is
        # the number of distinct prompts, and a sample-level AUC on it mostly measures
        # "did the easy prompts happen to be sampled more" -- see the per-prompt table
        # below. V_last is the value after reading the whole response and is what PPO
        # actually differences against.
        print("\n  AUC by feature (0.5 = no discrimination):")
        for key, label in (("v0", "V(s_0)  prompt only"), ("v_mean", "V_mean"), ("v_last", "V_last  full response")):
            pos = [r[key] for r in records if r["correct"] is True]
            neg = [r[key] for r in records if r["correct"] is False]
            a = auc(pos, neg)
            if a is not None:
                print(
                    f"    {label:22s} AUC={a:.4f}   mean|correct={statistics.mean(pos):.4f} "
                    f"mean|wrong={statistics.mean(neg):.4f}"
                )
        # Length as a baseline: if V_last cannot beat "shorter answers are right", the
        # head is not adding anything a two-line heuristic would not give.
        pos_len = [-r["n_response"] for r in records if r["correct"] is True]
        neg_len = [-r["n_response"] for r in records if r["correct"] is False]
        a_len = auc(pos_len, neg_len)
        if a_len is not None:
            print(f"    {'(baseline: -length)':22s} AUC={a_len:.4f}   <- V_last should beat this")

        _report_per_prompt(records)

        acc = len(v0_correct) / (len(v0_correct) + len(v0_wrong))
        print(f"\n  actual accuracy {acc:.4f}   vs mean V(s_0) {statistics.mean(all_v0):.4f}")
        print("    (CRITIC_EXCLUDE_OLP=1 makes V(s_0) an estimate of P(correct), so these")
        print("     two should be close if the critic is calibrated)")
    else:
        print("  没有对错标注（需要 'correct' 或 'label' 字段），跳过区分度检查")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--critic", required=True, help="critic dist ckpt, e.g. runs/<TAG>_critic/iter_0000499")
    ap.add_argument("--base", default=None, help="base HF dir, for config + tokenizer (needed unless --inspect)")
    ap.add_argument("--samples", default=None, help="jsonl with prompt/response [+ correct|label]")
    ap.add_argument("--join-data", default=None, help="original jsonl, to join prompts into eval.py --out records")
    ap.add_argument("--inspect", action="store_true", help="only report the value head; do not load the backbone")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=None, help="score only the first N samples")
    ap.add_argument("--model-name", default=None, help="override the converter dispatch key")
    ap.add_argument("--out", default=None, help="write per-sample values to this jsonl")
    args = ap.parse_args()

    # `.metadata` (a dotfile) + the `__*.distcp` shards ARE the checkpoint; common.pt
    # is optional and megatron does not write it here. Check for the real thing, or a
    # parent save dir gets rejected for the wrong reason while a genuine checkpoint
    # gets rejected for no reason at all.
    critic = Path(args.critic)
    if not (critic / ".metadata").is_file():
        candidates = sorted(p.name for p in critic.glob("iter_*") if (p / ".metadata").is_file())
        hint = f" Did you mean {critic / candidates[-1]}?" if candidates else ""
        raise SystemExit(f"{critic}/.metadata not found — not a dist checkpoint dir.{hint}")

    print("=" * 60)
    print(f"critic: {critic}")
    print("=" * 60)
    state_dict, megatron_args = load_dist_state_dict(critic)
    inspect_value_head(state_dict, megatron_args)

    if args.inspect or not args.samples:
        if not args.inspect:
            print("\n（没给 --samples，只做了 value head 检查。加 --samples 才会真跑前向。）")
        return
    if not args.base:
        raise SystemExit("--samples needs --base (the HF dir supplying config + tokenizer)")
    run_samples(args, state_dict, megatron_args)


if __name__ == "__main__":
    main()
