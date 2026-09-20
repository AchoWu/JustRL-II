#!/bin/bash
# JustRL2 正式训练 —— 1 节点 8 GPU / 32k / 500 步
#
#   bash run_train.sh                 # 首次启动，或中断后原地续跑（默认 16k）
#   bash run_train.sh --32k           # 32k（原配方的 1 节点版本；本机 KV cache 会抖动）
#   bash run_train.sh --16k           # 16k 变体（KV cache 装得下，但不是配方复现）
#   bash run_train.sh --probe         # 只跑 3 步实测速度和显存（独立 EXP_TAG，不污染正式 run）
#   bash run_train.sh --dry-run       # 只打印将要执行的命令和预检结果，不启动
#
# 权重默认会先拷到 /dev/shm/llms 再加载（tmpfs，消除网络存储的读盘开销）：
#   SHM_WEIGHTS=0 关掉，SHM_DIR=<path> 换位置。首次拷贝约需一次加载的时间。
#
# 与 run_test.sh 的区别：那个是冒烟（3 步 / GBS 16 / 2k 响应），只验证管道能转；
# 这个是真跑，全部参数取自 justrl2/configs/1node-8gpu-32k-baremetal.env 的默认值，
# 本文件不覆盖任何配方旋钮。
#
# 中断后重跑同一条命令即可：actor/critic checkpoint 会自动接上。
set -eo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40092/howu/test_gpu.py
' EXIT INT TERM


cd "$(dirname "${BASH_SOURCE[0]}")"

MODE=run
DRYRUN=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --probe)   MODE=probe ;;
    --dry-run) DRYRUN=1 ;;
    --16k)     CONFIG_PICK=justrl2/configs/1node-8gpu-16k-baremetal.env ;;
    --32k)     CONFIG_PICK=justrl2/configs/1node-8gpu-32k-baremetal.env ;;
    *)         ARGS+=("$arg") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# justrl2/train.sh 会调 setup.sh，它在 SKIP_PIP_INSTALL != 1 时会删掉所有 pip.conf、
# 把 index-url 改成 pypi.org、再装一批包 —— 内网机器上会卡住并破坏 pip 源配置。
# baremetal 配置里也设了一份，这里再设一次防止配置未同步。
export SKIP_PIP_INSTALL=1

# 显式的 CONFIG 环境变量优先于 --16k/--32k，两者都没给时用 16k。
CONFIG=${CONFIG:-${CONFIG_PICK:-justrl2/configs/1node-8gpu-16k-baremetal.env}}

# colocate 模式下 miles 会强制开 sglang 的 memory saver（卸载 rollout 权重用），
# 而 prefill 的 cuda graph backend 默认是 breakable，两者互斥：
#   NotImplementedError: Breakable CUDA graph is not compatible with memory saver mode
# 关掉 prefill 的 graph 最省事：decode 的 graph 不受影响（吞吐大头在那）。
# 注意这不是 run_test.sh 的默认值自动继承 —— 正式跑必须自己带上。
EXTRA_ARGS=${EXTRA_ARGS:---sglang-disable-prefill-cuda-graph}

# ---- probe 模式：先用 3 步实测，再决定 NUM_ROLLOUT -------------------------
# 冒烟是 2k 响应 / GBS 16，正式是 30720 响应 / GBS 64 —— token 量约 60 倍，且 32k
# 的 attention 是平方增长，墙钟时间无法从冒烟外推，必须实测。
# EXP_TAG 故意不同：否则这几步的 checkpoint 会落进正式 run 的目录，污染之后的 resume。
if [ "$MODE" = probe ]; then
  export EXP_TAG=${EXP_TAG:-probe32k}
  export NUM_ROLLOUT=3
  export SAVE_INTERVAL=1000       # 不存 ckpt
  export HF_SAVE_INTERVAL=0
  export EVAL_INTERVAL=1000       # 不做训练中 eval
  # train.sh 断言 CRITIC_LR_WARMUP_ITERS < TRAIN_ITERS，配方默认 10 > 3 会直接失败。
  # 那个断言在 8 个 engine 起完之后才抛，白等二十多分钟，所以这里必须一起改。
  export CRITIC_LR_WARMUP_ITERS=1
  # 配方默认 30 步 critic-only，3 步探针里 actor 永远不会更新 —— 而 actor 的
  # forward/backward 正是显存大头，探不到就失去意义。
  export NUM_CRITIC_ONLY_STEPS=1
fi

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG=${LOG:-$LOG_DIR/${MODE}_$(date +%m%d_%H%M%S).log}

# ---- 启动前预检（都是过去真踩过、且失败代价是几十分钟的坑）----------------
echo "============================================================"
echo "启动前检查"
echo "============================================================"
FATAL=0
note() { echo "  $1"; }
fail() { echo "  !!   $1"; FATAL=1; }

# 0) 权重放 tmpfs（/dev/shm）以加快加载。放在最前：它会改写 HF_MODEL_DIR /
#    MEGATRON_MODEL_PATH，后面的检查和 train.sh 都要看到改写后的值。
# models/ 在网络存储上，实测两处加载都很慢：
#   SGLang 读 HF 权重     8m42s（8 个引擎并发读同一份，I/O 争抢）
#   Megatron 读 dist ckpt 18m19s
# tmpfs 是内存文件系统，消除这两处的读盘开销。首次 cp 仍要走一遍慢网络（约等于
# 一次加载的时间），之后每次启动都省。
#
# SHM_WEIGHTS=0 关掉；SHM_DIR 换位置。
# 注意 /dev/shm 占的是**内存**：上一轮 after_offload_train 已用 223GB 主机内存，
# 加上权重十几 GB 没问题，但这部分不会自动释放，要手动 rm 或重启才回收。
# Ray 的 plasma object store（RAY_OBJECT_STORE_MEMORY，默认 4GB）也住在 /dev/shm，
# 所以容量检查要把它算进去。
SHM_DIR=${SHM_DIR:-/dev/shm/llms}
if [ "${SHM_WEIGHTS:-1}" = 1 ] && [ -d /dev/shm ]; then
  # 源路径：优先已 export 的值，否则取配置里的默认（配置用 : ${VAR:=...}，这里
  # 只读不写，用子 shell 隔离，避免污染当前环境）
  _src_models=$(bash -c "source '$CONFIG' >/dev/null 2>&1; echo \$MODELS_DIR" 2>/dev/null)
  _src_models=${MODELS_DIR:-${_src_models:-$PWD/models}}
  _src_hf=$(bash -c "source '$CONFIG' >/dev/null 2>&1; echo \$HF_MODEL_DIR" 2>/dev/null)
  _src_hf=${HF_MODEL_DIR:-${_src_hf:-$_src_models/JustRL-II-base-model}}
  _src_mg=$(bash -c "source '$CONFIG' >/dev/null 2>&1; echo \$MEGATRON_MODEL_PATH" 2>/dev/null)
  _src_mg=${MEGATRON_MODEL_PATH:-${_src_mg:-$_src_models/JustRL-II-base-model-torch_dist}}

  if [ ! -d "$_src_hf" ] || [ ! -d "$_src_mg" ]; then
    echo "  ??   tmpfs 加速跳过：源权重不存在（$_src_hf / $_src_mg）"
  else
    _need_mb=$(du -xsm "$_src_hf" "$_src_mg" 2>/dev/null | awk '{s+=$1} END {print s+0}' || echo 0)
    # 已经在 tmpfs 里的部分不用重复计入
    _have_mb=$(du -xsm "$SHM_DIR" 2>/dev/null | awk '{print $1+0}' || echo 0)
    _free_mb=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $4+0}')
    # 给 Ray 的 plasma store 留出余量（默认 4GB），再加 2GB 缓冲
    _reserve_mb=$(( (${RAY_OBJECT_STORE_MEMORY:-4000000000} / 1048576) + 2048 ))
    if [ -z "$_free_mb" ]; then
      echo "  ??   tmpfs 加速跳过：读不到 /dev/shm 容量"
    elif [ $((_need_mb - _have_mb + _reserve_mb)) -gt "$_free_mb" ]; then
      echo "  ??   tmpfs 加速跳过：/dev/shm 空闲 ${_free_mb}MB，需要 $((_need_mb - _have_mb))MB"
      echo "       + Ray plasma 预留 ${_reserve_mb}MB。用 SHM_WEIGHTS=0 显式关闭可消除本提示。"
    else
      mkdir -p "$SHM_DIR"
      for _pair in "$_src_hf" "$_src_mg"; do
        _dst="$SHM_DIR/$(basename "$_pair")"
        # 幂等：已存在且源没更新过就跳过。-u 只拷更新的文件，中断后重跑能续。
        if [ -d "$_dst" ] && [ -z "$(find "$_pair" -newer "$_dst" -print -quit 2>/dev/null)" ]; then
          echo "  OK   tmpfs 已有 $(basename "$_pair")（跳过拷贝）"
        else
          echo "  ..   拷到 tmpfs: $(basename "$_pair") -> $_dst（首次约需一次加载的时间）"
          mkdir -p "$_dst"
          cp -ru "$_pair"/. "$_dst"/ || { echo "  !!   拷贝失败，回退到原路径"; rm -rf "$_dst"; }
        fi
      done
      # 只在两个目录都就位时才切换，避免一半在 tmpfs 一半在网络存储
      if [ -d "$SHM_DIR/$(basename "$_src_hf")" ] && [ -d "$SHM_DIR/$(basename "$_src_mg")" ]; then
        export HF_MODEL_DIR="$SHM_DIR/$(basename "$_src_hf")"
        export MEGATRON_MODEL_PATH="$SHM_DIR/$(basename "$_src_mg")"
        echo "  OK   权重走 tmpfs: $SHM_DIR"
      fi
    fi
  fi
fi


# 1) cuDNN 必须只有一套。conda 装的那份（为了 cudnn.h）和 pip 的 9.22 同时在进程里
#    时，SONAME 都是 libcudnn.so.9，先载入的赢符号解析、子库版本却对不上，TE 的
#    FusedAttention 会在 critic 第一次前向报 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
#    —— 那是启动后 70 分钟。注意挪到 $CONDA_PREFIX/lib 的子目录**不够**，实测仍会被
#    载入，必须移出 prefix。见 justrl2/setup/bare_metal_cu129.sh。
cudnn_dirs=$(python - <<'PY' 2>/dev/null
import pathlib
try:
    import torch, transformer_engine.pytorch  # noqa: F401
    libs = {pathlib.Path(l.split()[-1]) for l in open('/proc/self/maps')
            if 'libcudnn' in l and l.split()[-1].startswith('/')}
    print('\n'.join(sorted({str(p.parent) for p in libs})))
except Exception as e:
    print(f'ERROR {e}')
PY
)
n_cudnn=$(echo "$cudnn_dirs" | grep -c '^/' || true)
if echo "$cudnn_dirs" | grep -q '^ERROR'; then
  note "??   cuDNN 来源检查跳过（$(echo "$cudnn_dirs" | head -1)）"
elif [ "$n_cudnn" -gt 1 ]; then
  fail "cuDNN 有 $n_cudnn 个来源目录，必然混载 —— 会在 critic 首次前向炸："
  echo "$cudnn_dirs" | sed 's/^/         /'
  echo "       修复: mkdir -p /root/cudnn_conda_backup && mv \$CONDA_PREFIX/lib/libcudnn*.so* /root/cudnn_conda_backup/"
else
  note "OK   cuDNN 单一来源"
fi

# 2) 本仓库对 ReloadableProcessGroup 的修复必须在位。缺了会在 critic 第一次梯度
#    规约时四个 rank 同时 SIGSEGV（PyWorkHolder::wait 空指针解引用）。
if grep -q "_CompletedWork" miles/utils/reloadable_process_group.py 2>/dev/null; then
  note "OK   ReloadableProcessGroup 的 None-Work 修复在位"
else
  fail "miles/utils/reloadable_process_group.py 缺少 _CompletedWork —— 先 git pull"
fi

# 3) 模型 / 数据存在。train.sh 也查，但它在 setup.sh 之后才查。
#    查的是解析后的路径而不是写死 models/ —— tmpfs 加速会把它们指到 /dev/shm。
for d in "${HF_MODEL_DIR:-models}" "${MEGATRON_MODEL_PATH:-models}" datasets; do
  [ -d "$d" ] || fail "$d 不存在（见 justrl2/prepare_model.sh / prepare_data.py）"
done

# 4) 磁盘余量。SAVE_INTERVAL=25 + HF_SAVE_INTERVAL=25，500 步各 20 份；Megatron 的
#    dist ckpt 靠 SAVE_RETAIN_INTERVAL 轮转只留最新一份（~63GB），但 HF 导出**不轮转**，
#    20 份 × ~5GB 会一直堆着。
avail_gb=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")
if [ -n "$avail_gb" ]; then
  if [ "$avail_gb" -lt 200 ]; then
    fail "当前盘剩余 ${avail_gb}GB —— dist ckpt ~63GB + HF 导出 20×5GB，建议 >200GB"
    echo "       或调大 HF_SAVE_INTERVAL（HF 导出随时能从 dist ckpt 重新生成）"
  else
    note "OK   磁盘剩余 ${avail_gb}GB"
  fi
fi

# 5) core dump 堆积。/dockerdata 不在 disk-watchdog 的监控范围（它只看 /tmp 和 /data）。
core_mb=$(du -xsm /dockerdata/core-* 2>/dev/null | awk '{s+=$1} END {print s+0}' || echo 0)
if [ "${core_mb:-0}" -gt 1024 ] 2>/dev/null; then
  note "??   /dockerdata 有 $((core_mb / 1024))GB core dump，watchdog 看不到："
  note "     rm -f /dockerdata/core-ray::MegatronTr-*"
fi

# 6) 续跑状态。半份 checkpoint 是硬错误，train.sh 会拒绝启动 —— 提前说清楚。
SAVE_ROOT_GUESS=${SAVE_ROOT:-$PWD/runs}
# EXP_TAG 由配置文件决定（--16k / --32k 各有自己的），所以从配置里读出来而不是写死 ——
# 否则 --16k 时会去查 32k 的目录，把「全新启动」误报成「续跑」。用子 shell 隔离，
# 避免这些默认值污染当前环境（train.sh 自己会再 source 一次）。
TAG_GUESS=${EXP_TAG:-$(bash -c "source '$CONFIG' >/dev/null 2>&1; echo \$EXP_TAG" 2>/dev/null)}
TAG_GUESS=${TAG_GUESS:-justrl2_minicpm5_2b_math32k_1node_baremetal}
A="$SAVE_ROOT_GUESS/$TAG_GUESS/latest_checkpointed_iteration.txt"
C="$SAVE_ROOT_GUESS/${TAG_GUESS}_critic/latest_checkpointed_iteration.txt"
if [ -f "$A" ] && [ -f "$C" ]; then
  note "OK   续跑: actor@$(cat "$A") / critic@$(cat "$C")"
  note "     若首次 resume 时改过 NUM_ROLLOUT，需加 OVERRIDE_OPT_PARAM_SCHEDULER=1"
elif [ -f "$A" ] || [ -f "$C" ]; then
  fail "半份 checkpoint（actor=$([ -f "$A" ] && echo 有 || echo 无) critic=$([ -f "$C" ] && echo 有 || echo 无)）—— train.sh 会拒绝启动"
else
  note "OK   全新启动（critic 从 base model 开始，value head 会被重新初始化）"
fi

# 7) Ray 残留。上一轮若是 segfault 死的，raylet 里有残留 worker。
if command -v pgrep >/dev/null 2>&1 && pgrep -f "raylet" >/dev/null 2>&1; then
  note "??   检测到 raylet 在跑，建议先: ray stop --force"
fi

echo
echo "config=$CONFIG"
echo "extra=$EXTRA_ARGS"
echo "mode=$MODE  log=$LOG"
[ "$MODE" = probe ] && echo "probe: NUM_ROLLOUT=3 NUM_CRITIC_ONLY_STEPS=1 EXP_TAG=$EXP_TAG（不存 ckpt）"
echo

[ "$FATAL" = 1 ] && { echo "有 FATAL 项，已中止。"; exit 1; }
if [ "$DRYRUN" = 1 ]; then
  echo "--dry-run: 将执行"
  echo "  bash justrl2/train.sh $CONFIG $EXTRA_ARGS $*"
  exit 0
fi

# ---- 启动 ------------------------------------------------------------------
bash justrl2/train.sh "$CONFIG" $EXTRA_ARGS "$@" 2>&1 | tee "$LOG"

# ---- 跑完后的检查 ----------------------------------------------------------
echo
echo "============================================================"
echo "配置生效检查"
echo "============================================================"
check() {  # check <期望字符串> <说明>
  if grep -qF -- "$1" "$LOG"; then echo "  OK   $2: $1"
  else echo "  !!   $2 未生效，实际命令行里没有 '$1'"; fi
}
check "--attention-backend fused"            "Megatron attention (无可用 flash-attn)"
check "--sglang-attention-backend flashinfer" "SGLang attention (fa3 在 driver 535 上会 CUDA error 222)"
check '"NVTE_FUSED_ATTN": "1"'               "TE cuDNN fused attention"
check "--no-gradient-accumulation-fusion"    "apex fused wgrad 已关"

echo
echo "============================================================"
echo "启动检查"
echo "============================================================"

# 1) value head 重新初始化。没有这条 run 就是静默坏的：head 会继承 LM head
#    重叠区的污染值，而训练看起来一切正常。续跑时不会有（finetune=False，保留已训练的 head）。
if grep -q "re-zeroed" "$LOG"; then
  grep -n "critic-value-head" "$LOG" | head -5
else
  echo "  ??   没有 [critic-value-head] ... re-zeroed"
  echo "       全新启动时这是致命的（value head 未重新初始化，run 无效）；续跑时正常。"
fi

# 2) critic 目标是否剔除了 overlong penalty（CRITIC_EXCLUDE_OLP=1）
grep -n "critic exclude shaping" "$LOG" | head -2 \
  || echo "  ??   没有 'critic exclude shaping'，检查 CRITIC_EXCLUDE_OLP"

# 3) 今天修掉的两个故障不该再出现
for pat in "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED" "Segfault encountered"; do
  if grep -q "$pat" "$LOG"; then echo "  !!   出现了 '$pat' —— 见 git log 里对应的修复 commit"; fi
done

# 4) 实测每步耗时 —— probe 模式的主要产出，用来外推 NUM_ROLLOUT 的墙钟时间
echo
echo "--- 每个 rollout 的耗时（generate_start -> train_end）---"
python - "$LOG" <<'PY' 2>/dev/null || echo "  ??   没解析到 VRAM-Phase 时间戳"
import re, sys
t = {}
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'VRAM-Phase rollout=(\d+) phase=(\w+) t=([\d.]+)', line)
    if m:
        t.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
for rid in sorted(t):
    p = t[rid]
    if 'generate_start' in p and 'train_end' in p:
        d = p['train_end'] - p['generate_start']
        gen = p.get('generate_end', 0) - p['generate_start'] if 'generate_end' in p else None
        extra = f"（生成 {gen/60:.1f} min）" if gen else ""
        print(f"  rollout {rid}: {d/60:.1f} min{extra}")
if len(t) >= 2:
    ds = [t[r]['train_end'] - t[r]['generate_start'] for r in sorted(t)
          if 'generate_start' in t[r] and 'train_end' in t[r]]
    if ds:
        avg = sum(ds) / len(ds)
        print(f"\n  平均 {avg/60:.1f} min/步 -> 500 步约 {avg*500/3600:.1f} 小时（不含启动的 ~25 min 权重加载）")
PY

# 5) 第一个 rollout 的采样文本必须通顺。乱码说明 MEGATRON_MODEL_PATH 指到了
#    iter_xxx 子目录而不是父目录 —— Megatron 会静默从随机权重开始。
echo
echo "--- 采样响应（应为通顺文本）---"
grep -n -m1 -oE "First rollout sample: .{0,600}" "$LOG" \
  || echo "  ??   没匹配到采样输出，手动检查：less $LOG"
