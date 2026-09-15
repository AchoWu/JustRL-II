#!/bin/bash
# JustRL2 冒烟测试 —— 最小规模跑通 PPO 全链路（不是缩小版配方，只验证管道）
#
#   bash run_train.sh
#
# 3 个 rollout、2 步 critic-only、GBS 16、2k 响应长度。目的是让 actor + critic +
# SGLang + Megatron 全部转一遍，几分钟内暴露问题；不要指望有任何训练效果。
#
# 真跑（500 步、32k）用：
#   bash justrl2/train.sh justrl2/configs/1node-8gpu-32k-baremetal.env
#
# 命名说明：与 justrl2/train.sh（真正的启动器）区分，这里只是冒烟包装。
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# justrl2/train.sh 会调 setup.sh，它在 SKIP_PIP_INSTALL != 1 时会删掉所有 pip.conf、
# 把 index-url 改成 pypi.org、再装一批包 —— 内网机器上会卡住并破坏 pip 源配置。
# baremetal 配置里也设了一份，这里再设一次防止配置未同步。
export SKIP_PIP_INSTALL=1

# 必须 export：justrl2/train.sh 是 bash 子进程，普通 shell 变量传不过去。
export EXP_TAG=smoke
export NUM_ROLLOUT=3
export NUM_CRITIC_ONLY_STEPS=2          # 前 2 步只训 critic，第 3 步才动 policy
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES_PER_PROMPT=4           # GBS = 4x4 = 16, actor DP = 4/1/2 = 2, 16%2=0 ✓
export OVER_SAMPLING_BATCH_SIZE=8
export ROLLOUT_MAX_RESPONSE_LEN=2048
export ROLLOUT_MAX_CONTEXT_LEN=4096
export OVERLONG_BUFFER_LEN=512          # 响应长度的 25%，DAPO soft overlong
export SAVE_INTERVAL=1000               # 冒烟不存 ckpt
export HF_SAVE_INTERVAL=0
export EVAL_INTERVAL=1000               # 不做训练中 eval

# LR warmup 必须跟着缩。Megatron 的 OptimizerParamScheduler 断言
#   lr_warmup_steps < lr_decay_steps
# 而两者都是按 GBS 换算的（miles/backends/megatron_utils/model.py）：
#   train_iters      = NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / GBS
#   lr_decay_steps   = train_iters * GBS
#   lr_warmup_steps  = CRITIC_LR_WARMUP_ITERS * GBS      （critic 也走这个函数）
# 配方默认 NUM_ROLLOUT=500 / CRITIC_LR_WARMUP_ITERS=10 是 10 < 500，没问题；
# 冒烟把 NUM_ROLLOUT 降到 3 后 train_iters=3，10 > 3 就直接 AssertionError
# —— 在 MegatronTrainRayActor.init() 里抛，8 个 SGLang engine 已经起完、
# Triton kernel 也编译完了才炸，白等二十多分钟，所以这里必须一起改。
export CRITIC_LR_WARMUP_ITERS=1

CONFIG=${CONFIG:-justrl2/configs/1node-8gpu-32k-baremetal.env}
LOG=${LOG:-/tmp/smoke_$(date +%m%d_%H%M%S).log}

# colocate 模式下 miles 会强制开 sglang 的 memory saver（卸载 rollout 权重用），
# 而 prefill 的 cuda graph backend 默认是 breakable，两者互斥：
#   NotImplementedError: Breakable CUDA graph is not compatible with memory saver mode
# prefill 可选 backend 只有 breakable / tc_piecewise / disabled（full 仅限 decode）。
# 关掉 prefill 的 graph 最省事：decode 的 graph 不受影响（吞吐大头在那），
# prefill 少一点速度，不影响正确性。想保留可试 tc_piecewise（走 torch.compile）。
EXTRA_ARGS=${EXTRA_ARGS:---sglang-disable-prefill-cuda-graph}

echo "config=$CONFIG  log=$LOG"
echo "extra=$EXTRA_ARGS"
bash justrl2/train.sh "$CONFIG" $EXTRA_ARGS "$@" 2>&1 | tee "$LOG"

# ---- 配置是否真的生效（三处 bare-metal 覆盖）--------------------------------
echo
echo "============================================================"
echo "配置生效检查"
echo "============================================================"
check() {  # check <期望字符串> <说明>
  if grep -qF -- "$1" "$LOG"; then echo "  OK   $2: $1"
  else echo "  !!   $2 未生效，实际命令行里没有 '$1'"; fi
}
check "--attention-backend fused"           "Megatron attention (无可用 flash-attn)"
check "--sglang-attention-backend flashinfer" "SGLang attention (fa3 cubin 会 JIT)"
check '"NVTE_FUSED_ATTN": "1"'              "TE cuDNN fused attention"
check "--no-gradient-accumulation-fusion"   "apex fused wgrad 已关"

# ---- 三条必查启动日志（见 CLAUDE.md / docs/reproduce.md）--------------------
echo
echo "============================================================"
echo "启动检查"
echo "============================================================"

# 1) value head 重新初始化。没有这条 run 就是静默坏的：head 会继承 LM head
#    重叠区的污染值，而训练看起来一切正常。
if grep -q "re-zeroed" "$LOG"; then
  grep -n "critic-value-head" "$LOG" | head -5
else
  echo "  !!   没有 [critic-value-head] ... re-zeroed —— value head 未重新初始化，run 无效"
fi

# 2) critic 目标是否剔除了 overlong penalty（CRITIC_EXCLUDE_OLP=1）
grep -n "critic exclude shaping" "$LOG" | head -2 \
  || echo "  ??   没有 'critic exclude shaping'，检查 CRITIC_EXCLUDE_OLP"

# 3) 第一个 rollout 的采样文本必须通顺。乱码说明 MEGATRON_MODEL_PATH 指到了
#    iter_xxx 子目录而不是父目录 —— Megatron 会静默从随机权重开始。
#    下面的匹配是尽力而为，没命中就自己翻日志找第一个 rollout 的输出。
echo "--- 采样响应（应为通顺文本）---"
grep -n -m3 -A6 -iE "sample response|^response|generated text" "$LOG" | head -40 \
  || echo "  ??   没匹配到采样输出，手动检查：less $LOG"
