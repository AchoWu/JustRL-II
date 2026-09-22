"""Value alignment for `justrl2/test_critic.py`, which scores a critic straight from
its Megatron dist checkpoint (no HF export).

The arithmetic is trivial; the *alignment* is not, and it is the part that fails
silently. Training reads values off `logits[start - 1 : end - 1]`
(`training_utils/loss.py:get_responses`), so `V(s_t)` — the value of the state before
response token `t` — comes from the hidden state of the token *preceding* it. An
off-by-one still produces plausible-looking numbers in [0, 1]; only the invariant
pinned here distinguishes them:

    V(s_0) == head(backbone(prompt_only).last_hidden_state[-1])

because with causal attention the last prompt position cannot see the response at
all. `test_value_is_causal` pins the other half: `V(s_t)` must not move when a later
token changes.

These run on CPU with a tiny random Llama. triton/sglang are stubbed because
`megatron_to_hf.processors` imports its fp8/mxfp8 quantizers eagerly, even though
this path never quantizes.
"""

import importlib.machinery
import sys
import types

import pytest
import torch


def _stub(name, **attrs):
    mod = sys.modules.get(name) or types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(scope="module")
def critic_mod():
    if "triton" not in sys.modules:
        _stub(
            "triton",
            jit=lambda f: f,
            cdiv=lambda a, b: -(-a // b),
            Config=object,
            autotune=lambda *a, **k: (lambda f: f),
        )
        _stub("triton.language", constexpr=int)
    if "sglang" not in sys.modules:
        _stub("sglang")
        _stub("sglang.srt")
        _stub("sglang.srt.utils", MultiprocessingSerializer=object)
        _stub("sglang.srt.utils.patch_torch", monkey_patch_torch_reductions=lambda: None)
        _stub("sglang.srt.weight_sync")
        _stub("sglang.srt.weight_sync.tensor_bucket", FlattenedTensorBucket=object)
        _stub("sglang.srt.layers")
        _stub("sglang.srt.layers.quantization")
        _stub("sglang.srt.layers.quantization.fp8_utils", mxfp8_group_quantize=None)
        _stub("flashinfer", mxfp8_quantize=lambda *a, **k: None)
    try:
        import justrl2.test_critic as mod
    except ImportError as e:  # a transitive dep this CPU stub set does not cover
        pytest.skip(f"test_critic not importable on CPU: {e}")
    return mod


H, N_LAYERS, FFN, NQG, KVC, NAH, VOCAB = 32, 2, 64, 2, 8, 4, 128


class _Args:
    num_layers = N_LAYERS
    hidden_size = H
    ffn_hidden_size = FFN
    num_attention_heads = NAH
    num_query_groups = NQG
    kv_channels = KVC
    vocab_size = VOCAB
    num_experts = None
    q_lora_rank = None
    sglang_enable_ep_moe = False
    critic_value_bias_init = 0.52


def _critic_state_dict() -> dict:
    qkv = NQG * (NAH // NQG + 2) * KVC
    return {
        "embedding.word_embeddings.weight": torch.randn(VOCAB, H) * 0.02,
        "decoder.final_layernorm.weight": torch.ones(H),
        "decoder.layers.self_attention.linear_qkv.weight": torch.randn(N_LAYERS, qkv, H) * 0.02,
        "decoder.layers.self_attention.linear_proj.weight": torch.randn(N_LAYERS, H, H) * 0.02,
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight": torch.ones(N_LAYERS, H),
        "decoder.layers.mlp.linear_fc1.weight": torch.randn(N_LAYERS, 2 * FFN, H) * 0.02,
        "decoder.layers.mlp.linear_fc2.weight": torch.randn(N_LAYERS, H, FFN) * 0.02,
        "decoder.layers.mlp.linear_fc1.layer_norm_weight": torch.ones(N_LAYERS, H),
        "output_layer.weight": torch.randn(1, H) * 0.05,
        "output_layer.bias": torch.full((1,), 0.52),
    }


@pytest.fixture(scope="module")
def built(critic_mod, tmp_path_factory):
    import json

    base = tmp_path_factory.mktemp("base")
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": H,
                "num_hidden_layers": N_LAYERS,
                "num_attention_heads": NAH,
                "num_key_value_heads": NQG,
                "head_dim": KVC,
                "intermediate_size": FFN,
                "vocab_size": VOCAB,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            }
        )
    )
    torch.manual_seed(0)
    sd = _critic_state_dict()
    backbone, head, _ = critic_mod.build_critic(sd, _Args, base, "llama", "cpu", torch.float32)
    return critic_mod, backbone, head, sd


def test_head_weights_come_from_the_checkpoint(built):
    _, _, head, sd = built

    assert torch.allclose(head.weight.float(), sd["output_layer.weight"].float())
    assert torch.allclose(head.bias.float(), sd["output_layer.bias"].float())


def test_values_are_one_per_response_token(built):
    mod, backbone, head, _ = built
    prompt_len, resp_len = 7, 5
    ids = torch.randint(0, VOCAB, (1, prompt_len + resp_len))

    values = mod.compute_values(backbone, head, ids, prompt_len)

    assert values.shape == (resp_len,)


def test_v_s0_equals_the_prompt_only_forward(built):
    """The invariant that distinguishes correct alignment from an off-by-one.

    V(s_0) is the value of the prompt before any response token exists, so it must
    equal the head at the last prompt position — and a prompt-only forward reaches
    that same position, because causal attention makes the prefix invariant.
    """
    mod, backbone, head, _ = built
    prompt_len, resp_len = 7, 5
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (1, prompt_len + resp_len))

    values = mod.compute_values(backbone, head, ids, prompt_len)
    with torch.no_grad():
        prompt_only = head(backbone(input_ids=ids[:, :prompt_len]).last_hidden_state).squeeze(-1).squeeze(0)

    assert torch.allclose(values[0], prompt_only[-1], atol=1e-4)
    # An off-by-one in either direction would land on a different position's value.
    assert not torch.allclose(values[0], prompt_only[-2], atol=1e-4)


def test_value_is_causal(built):
    """V(s_t) may not depend on tokens at or after t: changing the last token must
    leave every earlier value untouched. A slice shifted the other way would fail."""
    mod, backbone, head, _ = built
    prompt_len, resp_len = 7, 5
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (1, prompt_len + resp_len))
    perturbed = ids.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 13) % VOCAB

    values = mod.compute_values(backbone, head, ids, prompt_len)
    values_perturbed = mod.compute_values(backbone, head, perturbed, prompt_len)

    assert torch.allclose(values[: resp_len - 1], values_perturbed[: resp_len - 1], atol=1e-5)


def test_inspect_flags_an_untrained_head(critic_mod, capsys):
    """An all-zero weight is the CPU-backup bug's signature and nothing else reports it."""
    sd = _critic_state_dict()
    sd["output_layer.weight"] = torch.zeros(1, H)

    info = critic_mod.inspect_value_head(sd, _Args)

    assert info["nonzero"] == 0
    assert "ALL ZERO" in capsys.readouterr().out


def test_inspect_does_not_warn_on_a_bias_at_its_init(critic_mod, capsys):
    """A bias sitting at its init is the NORMAL case and must not warn.

    The bias is stored in bf16: one ulp at 0.52 is 2^-8 = 0.0039, and at
    critic_lr=5e-6 it takes ~390 same-sign steps to move one visible tick, so a
    healthy run prints 0.519531 for all 500 steps while the fp32 master does drift.
    Warning on that would fire on every run.
    """
    sd = _critic_state_dict()
    sd["output_layer.bias"] = torch.tensor([0.52], dtype=torch.bfloat16).float()
    sd["output_layer.weight"] = torch.randn(1, H) * 0.05  # a head with real spread

    critic_mod.inspect_value_head(sd, _Args)

    out = capsys.readouterr().out
    assert "??" not in out
    assert "!!" not in out


def test_inspect_flags_a_near_constant_head(critic_mod, capsys):
    """What is actually worth warning about: ||w||_2 too small to separate states,
    so V is effectively its prior and PPO's advantage degrades to whitened reward."""
    sd = _critic_state_dict()
    sd["output_layer.weight"] = torch.full((1, H), 1e-6)

    info = critic_mod.inspect_value_head(sd, _Args)

    assert info["est_v_spread"] < 0.05
    assert "varies by only" in capsys.readouterr().out


def test_est_v_spread_scales_with_the_weight_norm(critic_mod):
    """The estimate is ||w||_2 * rms(h), not the per-element rms: a head whose
    elements are individually tiny can still separate states across 2048 of them."""
    sd = _critic_state_dict()
    sd["output_layer.weight"] = torch.full((1, H), 1e-3)

    info = critic_mod.inspect_value_head(sd, _Args)

    expected_l2 = 1e-3 * H**0.5
    assert info["w_l2"] == pytest.approx(expected_l2, rel=1e-4)
    assert info["est_v_spread"] == pytest.approx(expected_l2 * critic_mod.HIDDEN_RMS_ESTIMATE, rel=1e-4)


def test_inspect_rejects_a_policy_checkpoint(critic_mod):
    sd = _critic_state_dict()
    sd["output_layer.weight"] = torch.randn(VOCAB, H)

    with pytest.raises(SystemExit, match="policy checkpoint"):
        critic_mod.inspect_value_head(sd, _Args)


def test_build_rejects_a_policy_checkpoint(critic_mod, tmp_path):
    """build_critic re-checks the shape rather than trusting an earlier inspect call:
    taking row 0 of a [vocab, H] LM head would yield plausible-looking non-values."""
    sd = _critic_state_dict()
    sd["output_layer.weight"] = torch.randn(VOCAB, H)

    with pytest.raises(SystemExit, match="policy checkpoint"):
        critic_mod.build_critic(sd, _Args, tmp_path, "llama", "cpu", torch.float32)


def test_backbone_conversion_drops_the_value_head(critic_mod):
    """The head must be held back before convert_to_hf sees it — that converter maps
    output_layer.weight to lm_head.weight and raises on output_layer.bias."""
    hf_tensors = critic_mod.convert_backbone(_critic_state_dict(), _Args, "llama")

    assert "lm_head.weight" not in hf_tensors
    assert not any("output_layer" in k for k in hf_tensors)
    assert "model.embed_tokens.weight" in hf_tensors
    for i in range(N_LAYERS):
        assert f"model.layers.{i}.self_attn.q_proj.weight" in hf_tensors


def test_args_rebuilt_from_hf_config_when_common_pt_is_absent(critic_mod, tmp_path):
    """megatron does not write common.pt for these training checkpoints, so the six
    fields the name mapping reads must come from the HF config instead. Requiring
    common.pt rejected a perfectly valid checkpoint."""
    import json

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": H,
                "num_hidden_layers": N_LAYERS,
                "num_attention_heads": NAH,
                "num_key_value_heads": NQG,
                "head_dim": KVC,
                "intermediate_size": FFN,
                "vocab_size": VOCAB,
                "rms_norm_eps": 1e-6,
            }
        )
    )

    rebuilt = critic_mod.megatron_args_from_hf_config(base, _critic_state_dict())

    # Exactly the fields get_named_params + the per-model converter read.
    assert rebuilt.num_layers == N_LAYERS
    assert rebuilt.hidden_size == H
    assert rebuilt.num_attention_heads == NAH
    assert rebuilt.num_query_groups == NQG
    assert rebuilt.kv_channels == KVC
    assert rebuilt.num_experts is None
    # and the conversion actually runs on them
    hf_tensors = critic_mod.convert_backbone(_critic_state_dict(), rebuilt, "llama")
    assert f"model.layers.{N_LAYERS - 1}.self_attn.q_proj.weight" in hf_tensors


def test_inspect_tolerates_missing_megatron_args(critic_mod, capsys):
    """Without common.pt the recipe knobs are unknown; the head report must still
    work rather than crash on getattr, and must say so instead of printing Nones."""
    info = critic_mod.inspect_value_head(_critic_state_dict(), None)

    assert info["nonzero"] == H
    assert info["bias"] == pytest.approx(0.52)
    assert info["gae_lambda_k"] is None
    out = capsys.readouterr().out
    assert "no common.pt" in out
    assert "gae_lambda_k" not in out


def test_repo_root_does_not_shadow_sglang(critic_mod, tmp_path):
    """`<repo>/sglang` is a source checkout whose package is at sglang/python/sglang.

    Prepending the repo root alone makes the bare `<repo>/sglang/` directory win as an
    implicit namespace package: `import sglang` yields `__file__ is None` and no
    `srt`, so every `sglang.srt.*` import under miles dies. The checkout's python/
    dir has to come first. Verified against a simulated layout in a subprocess,
    because the import is a module-level side effect.
    """
    import subprocess
    import textwrap

    (tmp_path / "sglang" / "python" / "sglang" / "srt").mkdir(parents=True)
    (tmp_path / "sglang" / "python" / "sglang" / "__init__.py").touch()
    (tmp_path / "sglang" / "python" / "sglang" / "srt" / "__init__.py").touch()

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        repo = Path(sys.argv[1])
        prepend = [repo / "sglang" / "python", repo]
        sys.path[:0] = [str(p) for p in prepend if p.is_dir()]
        import sglang, sglang.srt
        assert sglang.__file__ is not None, "shadowed by the bare <repo>/sglang dir"
        assert str(repo) in sys.path, "repo root must stay importable for miles/tools"
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_spearman_matches_known_values(critic_mod):
    assert critic_mod.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert critic_mod.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ties must get averaged ranks, or the coefficient is biased.
    assert critic_mod.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert critic_mod.spearman([1, 2], [1, 2]) is None  # n < 3
    assert critic_mod.spearman([1, 1, 1], [1, 2, 3]) is None  # zero variance


def test_per_prompt_grouping_recovers_the_real_sample_size(critic_mod, capsys):
    """V(s_0) is a function of the prompt, so N samples of one prompt share it exactly.

    Reported as 24 sample-level rows this looks like n=24; it is 3 prompts. The
    grouping is what stops a sample-level AUC on V(s_0) from being read as evidence.
    """
    records = [{"v0": 0.41, "v_mean": 0.5, "v_last": 0.6, "n_response": 100, "correct": True}] * 8
    records += [{"v0": 0.45, "v_mean": 0.5, "v_last": 0.5, "n_response": 100, "correct": True}] * 4
    records += [{"v0": 0.45, "v_mean": 0.5, "v_last": 0.5, "n_response": 100, "correct": False}] * 4
    records += [{"v0": 0.47, "v_mean": 0.5, "v_last": 0.4, "n_response": 100, "correct": False}] * 8

    critic_mod._report_per_prompt(records)

    out = capsys.readouterr().out
    assert "3 distinct prompts" in out
    # Perfectly anti-correlated: the lowest V(s_0) had the highest accuracy.
    assert "-1.000" in out
    # |rho| = 1 is strong ranking (inverted, but strong), so the "no signal" hint must
    # NOT fire here -- it is reserved for |rho| < 0.3, which is what the real 16k run
    # shows (rho = -0.26 over 9 prompts).
    assert "does not rank prompt difficulty" not in out


def test_per_prompt_flags_a_head_that_does_not_rank_difficulty(critic_mod, capsys):
    """The real 16k-run table: V(s_0) varies but barely correlates with accuracy.

    (V(s_0), n_samples, n_correct) as measured on AIME-2025, which gives rho = -0.26
    over 9 prompts -- inside the |rho| < 0.3 band the hint is meant to catch.
    """
    real = [
        (0.4160, 8, 8),
        (0.4395, 6, 4),
        (0.4414, 2, 1),
        (0.4531, 8, 7),
        (0.4551, 8, 8),
        (0.4590, 8, 8),
        (0.4648, 8, 1),
        (0.4727, 8, 8),
        (0.4785, 8, 0),
    ]
    records = []
    for v0, n, n_right in real:
        records += [{"v0": v0, "correct": True}] * n_right
        records += [{"v0": v0, "correct": False}] * (n - n_right)

    critic_mod._report_per_prompt(records)

    out = capsys.readouterr().out
    assert "9 distinct prompts" in out
    assert "-0.261" in out
    assert "does not rank prompt difficulty" in out


def test_per_prompt_skips_unlabelled_and_single_prompt(critic_mod, capsys):
    critic_mod._report_per_prompt([{"v0": 0.4, "correct": None}, {"v0": 0.5, "correct": None}])
    assert capsys.readouterr().out == ""

    critic_mod._report_per_prompt([{"v0": 0.4, "correct": True}, {"v0": 0.4, "correct": False}])
    assert capsys.readouterr().out == ""


def test_auc_ranks_correct_above_wrong(critic_mod):
    assert critic_mod.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert critic_mod.auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    # ties count as half, so a constant value head scores exactly 0.5 — which is the
    # number to watch for: it means V carries no signal about correctness.
    assert critic_mod.auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert critic_mod.auc([], [0.1]) is None
