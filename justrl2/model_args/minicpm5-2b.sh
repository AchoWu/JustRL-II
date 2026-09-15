# MiniCPM5-2B model args for Megatron (dense Llama architecture).
# Mirrors config.json of openbmb/JustRL-II-base-model, the RL initialization
# checkpoint of MiniCPM5-2B (openbmb/MiniCPM5-2B).
# Gradient accumulation fusion needs apex's fused_weight_gradient_mlp_cuda extension.
# When it is missing, Megatron's ColumnParallelLinear raises at construction time:
#   RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion
#   set to True but the custom CUDA extension ... is not found
# On the bare-metal cu12 stack the extension is ABI-broken against torch 2.13 (the same
# c10::impl::cow::materialize_cow_storage break as flash-attn), so it has to be off.
#
# Autodetect rather than relying on the caller: this file is sourced by both train.sh
# (which reads a config that can set NO_GRAD_ACC_FUSION) and prepare_model.sh (which
# reads no config at all), so a config-only knob silently left the converter broken.
# Setting NO_GRAD_ACC_FUSION explicitly still wins — export NO_GRAD_ACC_FUSION= to force
# fusion on even when the probe fails.
if [ -z "${NO_GRAD_ACC_FUSION+x}" ]; then
  if python -c "import fused_weight_gradient_mlp_cuda" 2>/dev/null; then
    NO_GRAD_ACC_FUSION=""
  else
    NO_GRAD_ACC_FUSION=1
    echo "[model_args] fused_weight_gradient_mlp_cuda unavailable -> --no-gradient-accumulation-fusion" >&2
  fi
fi

MODEL_ARGS=(
   --swiglu
   --disable-bias-linear
   --num-layers 42
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128
   --vocab-size 130560
   --make-vocab-size-divisible-by 64
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-5000000}"
   --max-position-embeddings 65536
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --untie-embeddings-and-output-weights
   --no-masked-softmax-fusion
   --no-rope-fusion
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   ${NO_GRAD_ACC_FUSION:+--no-gradient-accumulation-fusion}
)
