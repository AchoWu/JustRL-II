# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import logging
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from miles.utils.misc import load_function
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
        # This layer REPLACES model.output_layer after GPTModel.__init__, which is
        # where Megatron stamps is_embedding_or_output_parameter on the original
        # weight (language_module.py). Restore the flag on the replacement: muon
        # routes 2D-without-flag params into Newton-Schulz orthogonalization, whose
        # update magnitude is error-independent — for the critic's [1, H] value head
        # that means a fixed-size oscillation around the optimum instead of
        # convergence. With the flag, the head takes the matched-adamw branch like
        # every other embedding/output weight.
        self.weight.is_embedding_or_output_parameter = True

        if output_size == 1:
            # Scalar value head (critic): ZERO init, not normal(0, 0.02).
            # normal(0,0.02) on a hidden_size=2048 projection yields V with
            # std = 0.02*sqrt(H)*rms(h) ~ 9.7, while the value targets live in
            # [0, 1] — at critic_lr 5e-6 that offset needs ~90 optimizer steps to
            # close. Zero init makes V == 0 at step 0, so with
            # --normalize-advantages adv degrades exactly to whitened R (GRPO-like
            # failure-safe start) while the gradient still flows: dV/dw = h != 0.
            # NOTE: this only covers construction; the policy-ckpt load would still
            # overwrite the head with LM-head row 0 — checkpoint.py's
            # _rezero_critic_value_head handles that (plus the fp32 master resync).
            self.weight.data.zero_()
        else:
            self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            # JustRL2: with a zero weight V == bias at step 0, so the bias is the
            # critic's prior. Seeding it at the expected mean reward
            # (--critic-value-bias-init, 0.52 for the math training mix) makes the value
            # loss open at ~Var(r) instead of ~E[r^2] and keeps the first critic
            # gradient norm ~3x smaller; with 0 the head spends its first ~25 steps
            # learning the offset while the policy already updates against it.
            self.bias.data.fill_(float(bias_init) if output_size == 1 else 0.0)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        # Value-head diagnostic. The critic's weight gradient is exactly zero while
        # its bias gradient is not, and dV/dw = input_ while dV/db = 1, so the
        # hidden states arriving here are the remaining suspect (reproduced offline:
        # zero input gives exactly this signature, and nothing else does).
        #
        # Log the first few calls, not just one: the first forward of a rollout is
        # forward_only (get_values) under torch.no_grad, so a single sample cannot
        # tell us anything about the training forward -- which is the one whose
        # gradient we care about. grad_enabled distinguishes the two phases.
        if self.out_features == 1:
            # 分相位计数：forward_only（no_grad）和训练 forward 各留配额，否则前者
            # 会把额度用光 —— 实测 6 次全落在 no_grad 上，而我们要看的恰恰是训练那次。
            _phase = "train" if torch.is_grad_enabled() else "fwdonly"
            _counts = getattr(LinearForLastLayer, "_vh_input_counts", None)
            if _counts is None:
                _counts = {}
                LinearForLastLayer._vh_input_counts = _counts
            _n = _counts.get(_phase, 0)
            if _n < 3:
                _counts[_phase] = _n + 1
                _t = input_.detach()
                # 零输入已确认。现在要知道它是不是「整个张量恒零」还是只有部分位置零，
                # 以及 std —— 若 absmax=0 但 std>0 是不可能的，可用来排除采样错误。
                _f = _t.float()
                logger.info(
                    "[vh-diag] output_layer input [%s#%d]: shape=%s dtype=%s absmax=%.6g "
                    "std=%.6g nonzero=%d/%d requires_grad=%s is_leaf=%s grad_fn=%s",
                    _phase,
                    _n,
                    tuple(_t.shape),
                    _t.dtype,
                    float(_f.abs().max()),
                    float(_f.std()),
                    int((_t != 0).sum()),
                    _t.numel(),
                    input_.requires_grad,
                    input_.is_leaf,
                    type(input_.grad_fn).__name__ if input_.grad_fn is not None else "None",
                )
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def _attach_vh_input_probe(model, role: str) -> None:
    """Value-head diagnostic: sample what the final layernorm and decoder emit.

    Attached for BOTH roles. The critic's decoder output is all zero, but nothing
    has established whether the actor's is too -- its rollout text is generated by
    SGLang from a separate copy of the weights, so coherent samples say nothing
    about the megatron-side forward. If the actor is equally zero, the bug is
    global (and its log_probs, hence TIS and the KL term, are silently degenerate);
    if only the critic is, the difference is the head replacement itself.
    """

    def _make(tag):
        def _probe(_mod, _args_in, _out):
            _c = getattr(_probe, "_n", 0)
            if _c >= 2:
                return
            _probe._n = _c + 1
            _t = _out[0] if isinstance(_out, tuple) else _out
            if not torch.is_tensor(_t):
                return
            _f = _t.detach().float()
            logger.info(
                "[vh-diag] %s/%s [%s#%d]: shape=%s absmax=%.6g std=%.6g nonzero=%d/%d",
                role,
                tag,
                "train" if torch.is_grad_enabled() else "fwdonly",
                _c,
                tuple(_t.shape),
                float(_f.abs().max()),
                float(_f.std()),
                int((_t.detach() != 0).sum()),
                _t.numel(),
            )

        return _probe

    dec = getattr(model, "decoder", None)
    if dec is not None:
        dec.register_forward_hook(_make("decoder-out"))
        # final_layernorm is the last thing the block applies; if the block output is
        # zero but this is not, the zeroing happens in the block's tail.
        fln = getattr(dec, "final_layernorm", None)
        if fln is not None:
            fln.register_forward_hook(_make("final_layernorm"))
        # And the first layer: zero here means the input embedding or the very first
        # layer is the origin, not something accumulated across 42 layers.
        layers = getattr(dec, "layers", None)
        if layers is not None and len(layers) > 0:
            layers[0].register_forward_hook(_make("layer0-out"))
    emb = getattr(model, "embedding", None)
    if emb is not None:
        emb.register_forward_hook(_make("embedding-out"))


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size,
                    output_size=1,
                    config=model.config,
                    bias_init=getattr(args, "critic_value_bias_init", 0.0),
                )
            if post_process:
                _attach_vh_input_probe(model, role)
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.context_parallel_size = args.context_parallel_size
        provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        provider.variable_seq_lengths = args.variable_seq_lengths
        if hasattr(args, "moe_token_dispatcher_type"):
            provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if getattr(args, "moe_router_bias_update_rate", None) is not None:
            provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
        if getattr(args, "moe_aux_loss_coeff", None) is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
        provider.finalize()

        def wrapped_bridge_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            return provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        return wrapped_bridge_provider

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        config: TransformerConfig | None = None,
        pg_collection=None,
    ) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        assert config is None, "miles builds the config from args, so it expects config to be None"
        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Megatron's layer-spec signatures drift between releases (e.g.
                # moe_use_legacy_grouped_gemm and use_true_on_policy_backend were dropped,
                # and the matching --moe-use-legacy-grouped-gemm flag went with them).
                # Build the full kwarg set, then keep only what this checkout accepts.
                def _accepted(fn, **kwargs):
                    params = inspect.signature(fn).parameters
                    return {k: v for k, v in kwargs.items() if k in params}

                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        **_accepted(
                            get_gpt_layer_with_transformer_engine_spec,
                            num_experts=args.num_experts,
                            moe_grouped_gemm=args.moe_grouped_gemm,
                            qk_layernorm=args.qk_layernorm,
                            multi_latent_attention=args.multi_latent_attention,
                            moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
                        )
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        **_accepted(
                            get_gpt_layer_local_spec,
                            num_experts=args.num_experts,
                            moe_grouped_gemm=args.moe_grouped_gemm,
                            qk_layernorm=args.qk_layernorm,
                            multi_latent_attention=args.multi_latent_attention,
                            moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
                            normalization=args.normalization,
                            use_kitchen=config.use_kitchen,
                            use_true_on_policy_backend=config.true_on_policy_contract is not None,
                            use_kitchen_attention=config.use_kitchen_attention,
                            kitchen_attention_backend=config.kitchen_attention_backend,
                        )
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
            "rope_scaling_factor": getattr(args, "rope_scaling_factor", 8.0),
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            # hard code here to skip r3 registration for mtp layers
            # getattr is required to avoid ckpt conversion errors
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = False
                logger.warning(
                    "Rollout routing replay is not applicable for MTP modules, so skipped replay registration"
                )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = True

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(
                input_size=config.hidden_size,
                output_size=1,
                config=config,
                bias_init=getattr(args, "critic_value_bias_init", 0.0),
            )

        if post_process:
            _attach_vh_input_probe(model, role)

        return model

    return model_provider
