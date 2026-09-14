# MiniCPM5-2B model args for Megatron (dense Llama architecture).
# Mirrors config.json of openbmb/JustRL-II-base-model, the RL initialization
# checkpoint of MiniCPM5-2B (openbmb/MiniCPM5-2B).
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
   # Gradient accumulation fusion needs apex's fused_weight_gradient_mlp_cuda extension.
   # On the bare-metal cu12 stack that extension is ABI-broken against torch 2.13 (same
   # c10::impl::cow::materialize_cow_storage break as flash-attn), so set
   # NO_GRAD_ACC_FUSION=1 there. Costs some throughput, no correctness impact.
   ${NO_GRAD_ACC_FUSION:+--no-gradient-accumulation-fusion}
)
