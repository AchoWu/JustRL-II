#!/bin/bash
# ============================================================================
# JustRL2 裸机环境 — CUDA 12.9 变体
# ============================================================================
# ⚠️ 先看这里：这个脚本现在是最后手段，不是驱动装不上 580 时的默认路径。
#
# Miles 官方已经发布了 CUDA 12.9 镜像 radixark/miles:dev-cu12
# （即 `docker/build.py --variant cu12-x86`，基于 lmsysorg/sglang:v0.5.19-cu129），
# 里面 torch 2.13.0 / CUDA 12.9.2 / TE 2.17 / sgl-kernel 0.4.6.post1 / flashinfer
# 0.6.18 与本脚本手工装出来的完全是同一套，且 wheels 都来自同一个
# miles-wheels@cu129-x86_64 release。它的 NVIDIA_REQUIRE_CUDA 明确列了
# driver>=535，所以 535 驱动可以直接用：
#
#   docker build --build-arg MILES_IMAGE=radixark/miles:dev-cu12 -t justrl2 .
#
# 用镜像可以省掉下面每一节的坑：TE torch 扩展不用源码编译（镜像用配套 wheel 且构建期
# 自检）、flash-attn FA2+FA3 都能用（不必退 cuDNN fused attention）、apex 的
# fused wgrad 可用（不必 NO_GRAD_ACC_FUSION=1）、不必清 cu13 包、不必弄 libcuda stub。
# 也就是说用镜像时 justrl2/configs/1node-8gpu-32k-baremetal.env 里那三处覆盖都不需要，
# 直接跑 1node-8gpu-32k.env 即可。
#
# 只有在**连 docker 都用不了**（没权限 / 平台不给跑容器）时才走这个脚本。
# 下面的注释记录了每个坑的具体报错和成因，排查同类问题时仍有参考价值。
#
# 适用于：不能用任何 radixark/miles 镜像，且宿主机驱动无法升级到 580+ 的机器。
# 已在这台机器上全程走通：
#
#   8x NVIDIA H20 (sm90, 96GB) | driver 535.161.08 | glibc 2.38
#   Python 3.12.14 | torch 2.13.0+cu129 | nvcc 12.9 (conda)
#
# 为什么是 cu129 而不是 Miles 默认的 cu130：
#   Miles 官方 Dockerfile 基于 lmsysorg/sglang:v0.5.19 (CUDA 13.0.3)，而 CUDA 13.x
#   要求 driver >= 580，CUDA 12.x 只要求 >= 525。535 驱动因此只能走 cu12 变体。
#   副作用：535 最高只能 JIT 到 PTX ISA 8.2，而 CUDA 12.9 的 nvcc 生成 PTX 8.8。
#   带真实 sm90 SASS 的 kernel 正常跑；依赖运行时 PTX JIT 的会报 CUDA error 222
#   ("provided PTX was compiled with an unsupported toolchain")。所以
#   TORCH_CUDA_ARCH_LIST 只能是 9.0，绝不能写 9.0+PTX。
#
# 用法：
#   conda create -n justrl2 python=3.12 -y && conda activate justrl2
#   bash justrl2/setup/bare_metal_cu129.sh
#
# 顺序是有语义的，不要重排。逐段说明见各段注释。
# ============================================================================

WHEELS=https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64

# ---------------------------------------------------------------------------
# 1) torch —— 必须第一个装
# ---------------------------------------------------------------------------
# flashinfer_python 声明了 torch 依赖。torch 不在场时 pip 会从默认 PyPI 拉一个
# CUDA 13 构建的 torch（约 3GB），之后还得覆盖掉，纯浪费。
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu129

# ---------------------------------------------------------------------------
# 2) CUDA 12.9 toolkit + 环境变量 + libcuda stub
# ---------------------------------------------------------------------------
# 系统 /usr/local/cuda 是 12.4，而 flashinfer / TE / Megatron 的 JIT 会传
# --compress-mode=size（nvcc >= 12.8 才有的 flag），12.4 会报
# "nvcc fatal: Unknown option '--compress-mode=size'"。装进 conda 环境，不动系统的。
conda install -y -c nvidia cuda-toolkit=12.9

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
nvcc --version | grep -q "release 12.9" \
  || { echo "FATAL: nvcc 不是 12.9，检查 PATH 里是否被 /usr/local/cuda/bin 抢先"; exit 1; }

# 固化到 activate.d：新开 shell 若 CUDA_HOME 丢失，nvcc 会退回 12.4，
# 后面任何 JIT / 源码编译都会重演上面那个错。这一步是必需的，不是优化。
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda129.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
# 任何 pip install 都自动带上 constraints。PyPI 上 torch / nvidia-* 默认是 CUDA 13
# 构建，driver 535 用不了，而且直到 init_process_group 才报
# "CUDA driver version is insufficient" —— 失败点离安装很远，极难联想。
# 注意 torch 必须写成 2.13.0+cu129：PEP 440 下 2.13.0+cu130 也满足 ==2.13.0，
# 不带本地版本号的约束形同虚设。
export PIP_CONSTRAINT="$CONDA_PREFIX/justrl2_constraints.txt"
EOF

cat > "$CONDA_PREFIX/justrl2_constraints.txt" <<'EOF'
torch==2.13.0+cu129
torchvision==0.28.0
torchaudio==2.11.0
numpy<2
sglang-router<0.3.2
# requirements.txt 里 transformers 不带版本，pip 会装最新版，然后 sglang 启动时报
#   ValueError: 'qwen3_asr' is already used by a Transformers config
# —— 新版 transformers 已内置该 config，而 sglang-miles 仍在自己注册。
transformers==5.12.1
# 新版 kernels 要求 LayerRepository() 必传 revision/version，而 transformers 5.12.1
# 的 hub_kernels.py 不传，import 时就报
#   ValueError: Either a revision or a version must be specified.
# 这是 transformers 5.12.1 自己声明的兼容区间。
kernels>=0.12.0,<0.13
EOF

# flashinfer 的 JIT 链接时用 -L$CONDA_PREFIX/lib64 -L$CONDA_PREFIX/lib64/stubs -lcuda，
# 但 conda 把库放在 lib/ 而非 lib64/，该目录不存在 -> "ld: cannot find -lcuda"。
# 必须先 mkdir，否则 ln 报 "No such file or directory"。
# 链 stub 而非 /lib64/libcuda.so.1（真实驱动）：stub 只提供链接期符号，运行时由
# loader 找真驱动。且只能放 lib64/stubs/ 这个非默认搜索路径 —— stub 的 SONAME
# 也是 libcuda.so.1，落到 lib/ 会在运行时遮蔽真实驱动，故障极难定位。
mkdir -p "$CONDA_PREFIX/lib64/stubs"
ln -sf "$CONDA_PREFIX/targets/x86_64-linux/lib/stubs/libcuda.so" \
       "$CONDA_PREFIX/lib64/stubs/libcuda.so"
ls -lL "$CONDA_PREFIX/lib64/stubs/libcuda.so" >/dev/null \
  || { echo "FATAL: libcuda stub 链接断了"; exit 1; }

# ---------------------------------------------------------------------------
# 3) kernel 层
# ---------------------------------------------------------------------------
# PyPI 上的 sglang-kernel 0.4.6.post1 只有 CUDA 13 构建；带 +cu129 的本地版本号
# 只存在于 sgl-project/whl 的 GitHub release。--no-deps 防止 pip 从 PyPI 拖
# torch 覆盖第 1 步装好的 cu129 版本。注意包名是 sglang-kernel，模块名是 sgl_kernel。
pip install --no-deps \
  https://github.com/sgl-project/whl/releases/download/v0.4.6.post1/sglang_kernel-0.4.6.post1+cu129-cp310-abi3-manylinux2014_x86_64.whl

pip install "flashinfer_python[cu12]==0.6.18"
# jit-cache 提供预编译 sm90 cubin。没有它 flashinfer 会在首次调用时现场编译，
# 慢且在 8 个 rollout worker 间打架。
pip install flashinfer-jit-cache==0.6.18 --index-url https://flashinfer.ai/whl/cu129

# Miles pin 的 sglang_router 0.3.2 是 manylinux_2_39（要 glibc >= 2.39），
# 本机 glibc 2.38，pip 硬拒。requirements.txt 只要求 >=0.2.3，降版本即可；
# 单机 colocate rollout 基本不走 router 的逻辑。
pip install "sglang-router>=0.2.3,<0.3.2"

# pip torch==2.13.0 again
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129

# ---------------------------------------------------------------------------
# 4) TransformerEngine —— 元包 + CUDA 核心用预编译，torch 扩展必须源码重编
# ---------------------------------------------------------------------------
# 关键坑：miles-wheels 的 cu129-x86_64 tag 是 2026-07 构建的，针对 torch <= 2.12。
# torch 2.13.0 把 c10::impl::cow::materialize_cow_storage(StorageImpl&) 改成了
# materialize_cow(StorageImpl*)（commit d776d9d，pluggable MaterializeFn hook）。
# 旧版那个函数是头文件里的 inline 方法，会被烧进任何包含它的 .so，所以
# transformer_engine_torch 的预编译 wheel 在 torch 2.13 上 import 就炸：
#   undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
# miles-wheels 只为 cu130 出了 torch213 重建版（cu130-torch213-x86_64），
# cu129 没有对应 tag。所以这里跳过 transformer_engine_torch 的 wheel，改为源码编译。
#
# 元包和 CUDA 核心与 torch 无关，直接用预编译的。--no-deps 是必需的：
# 它们的 metadata 指向 cu13 runtime，不能让 pip 去解析。
pip install --no-deps "$WHEELS/transformer_engine-2.17.0-py3-none-any.whl"
pip install --no-deps "$WHEELS/transformer_engine_cu12-2.17.0-py3-none-manylinux_2_28_x86_64.whl"

# apex 同批构建。顶层 import 没问题（纯 Python），但它的 CUDA 扩展
# fused_weight_gradient_mlp_cuda 有和 flash-attn 一样的 torch 2.13 ABI 断裂，
# 只在真正用到时才暴露：
#   RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion
#   set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda ... not found
# 重编 apex 要编大量 .cu，不值得 —— 这只是把 wgrad 融进 GEMM 的性能优化。
# 跑 prepare_model.sh / train.sh 时带上 NO_GRAD_ACC_FUSION=1 关掉即可
# （见 justrl2/model_args/minicpm5-2b.sh），损失一点吞吐，不影响正确性。
pip install --no-deps "$WHEELS/apex-0.1-cp312-cp312-linux_x86_64.whl"

# TE 的 torch 扩展编译需要 nccl / cudnn / nvtx3 的开发头文件，conda 的
# cuda-toolkit 是运行时包不含这些，缺了会报：
#   fatal error: nccl.h / cudnn.h / nvtx3/nvToolsExt.h: No such file or directory
#
# nccl.h 不用额外装 —— torch 依赖的 nvidia-nccl-cu12 自带头文件，只要把它加进
# CPATH 就行。切勿 `conda install nccl`：nvidia channel 上是 NCCL 2.30.7，
# 需要比 535 更新的驱动，而且它的 $CONDA_PREFIX/lib/libnccl.so.2 会在
# init_process_group 时被 dlopen 抢先加载，遮蔽 torch 自带的 2.29.7，报
#   NCCL error ... unhandled cuda error, NCCL version 2.30.7
#   Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'
# 若已误装，把 lib/libnccl.so{,.2,.2.30.7} 挪出 lib/ 即可（头文件可留）。
conda install -y -c nvidia -c conda-forge cudnn nvtx-c
NCCL_INC="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/include"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$NCCL_INC${CPATH:+:$CPATH}"
[ -f "$NCCL_INC/nccl.h" ] || { echo "FATAL: 没找到 nccl.h，torch 的 nvidia-nccl-cu12 是否装上了？"; exit 1; }

# transformer_engine_torch 的 sdist 只有 33 个 .cpp、零个 .cu（CppExtension，
# 不重编 437MB 的 CUDA 核心），几分钟就好，不是一两小时。
#   --no-build-isolation : sdist 用 legacy backend，隔离环境里看不到已装的 torch
#   --no-deps            : sdist 硬依赖 transformer_engine_cu13，会覆盖上面的 cu12
#   NVTE_WITH_NCCL_EP=0  : NCCL expert-parallel 只对 MoE 有用，本模型是 dense Llama
#   NVTE_PYTORCH_FORCE_BUILD : 跳过预编译 wheel 探测（cu12+torch2.13 本来就 404）
MAX_JOBS=32 NVTE_FRAMEWORK=pytorch NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_WITH_NCCL_EP=0 \
  pip install --no-build-isolation --no-deps transformer_engine_torch==2.17.0

# ---------------------------------------------------------------------------
# 5) flash-attn —— 绝对不要装
# ---------------------------------------------------------------------------
# materialize_cow_storage 是 torch 2.2 引入、2.13 移除的，所以 torch 2.2~2.12 的
# 每一个 flash-attn wheel（含官方 Dao-AILab 的）在 torch 2.13 上都会以同样方式
# 失败 —— 已通过解析官方 flash_attn-2.8.3+cu12torch2.9 wheel 的 ELF .dynsym 确认。
# 官方最高只出到 cu12torch2.9，torch 2.10+ 只有 cu13（535 驱动用不了）。
# 源码编译要 1-3 小时（数百个 .cu 模板实例）。
#
# 而 TE 2.17 的 backends.py 是按包的 metadata 判断的，不是 NVTE_FLASH_ATTN：
#   try:    fa_utils.version = PkgVersion(get_pkg_version("flash-attn"))
#   except PackageNotFoundError: pass                   # 干净跳过
#   else:   from flash_attn_2_cuda import varlen_bwd    # 崩在这里
# 所以 NVTE_FLASH_ATTN=0 救不了（它在 get_attention_backend() 里才被读，太晚了），
# 只要包不存在，TE 就走 cuDNN FusedAttention（libtransformer_engine.so 里的 C++
# 后端，sm90 支持最好，且 context_parallel + fused_attention 是官方支持组合）。
#
# 配套的 train.sh 改动（两处都要，否则仍会走 flash 路径）：
#   --attention-backend fused      （原 flash）
#   "NVTE_FUSED_ATTN": "1"         （原 "0"）
#
# 副作用：miles 的 training_utils/loss.py 和 chunked_cross_entropy.py 会 import
# flash_attn.ops.triton.cross_entropy，但都有 try/except ImportError，降级到
# fp32 fallback。代价是 130k 词表的 chunked CE 保持 fp32 logits，多点显存，
# 不影响正确性（96GB 显存无所谓）。
pip uninstall -y flash-attn 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6) 框架 + miles 本体
# ---------------------------------------------------------------------------
# Megatron-LM 和 sglang 必须在仓库根目录旁（train.sh 第 28 行硬检查），
# PYTHONPATH 由 train.sh 设为 ".:Megatron-LM:sglang/python"。
#   Megatron-LM : radixark/Megatron-LM @ miles-main  （已验证 8c1e05747, core 0.19.0）
#   sglang      : sgl-project/sglang  @ sglang-miles （不是 radixark fork，那个 404）
# 不要打 third_party/patches/megatron.patch —— miles-main 本身就是打好补丁的 fork，
# 且 README 说的基线 nightly-dev-20260113 在两个仓库里都不存在这个 git ref。
[ -e Megatron-LM ] || { git clone --depth 1 -b miles-main \
    https://github.com/radixark/Megatron-LM.git _Megatron-LM && ln -sfn _Megatron-LM Megatron-LM; }
[ -e sglang ] || { git clone --depth 1 -b sglang-miles \
    https://github.com/sgl-project/sglang.git _sglang && ln -sfn _sglang sglang; }

# HF <-> Megatron 权重映射，tools/convert_hf_to_torch_dist.py 用
pip install --no-deps "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c"
pip install --no-deps "git+https://github.com/radixark/Megatron-Bridge.git@bridge"

pip install -e . --no-deps
pip install "math-verify==0.9.0" antlr4-python3-runtime
pip install "ray[default]>=2.56.0"
pip install absl-py          # 不装会在每次 megatron import 时刷 warning
pip install onnxscript
# Megatron 的异步 dist-checkpoint writer 依赖 psutil，缺了在 save_checkpoint 时才报
#   RuntimeError: Worker failure: ... psutil is not installed
pip install psutil
pip install pytest pytest-asyncio   # requirements.txt 只有 pytest-asyncio，缺 pytest 本体
# prepare_model.sh 第 23 行用 `hf download`（huggingface_hub 的新版 CLI，
# 不是旧的 huggingface-cli），不装会报 "hf: command not found"。
# hf_transfer 是 Rust 多线程下载，几十 GB 权重快数倍，配合 HF_HUB_ENABLE_HF_TRANSFER=1。
pip install -U "huggingface_hub[cli,hf_transfer]"
pip install mbridge datasets transformers
# ---------------------------------------------------------------------------
# 7) 清掉混进来的 cu13 包 —— 必须在 numpy 之前
# ---------------------------------------------------------------------------
# 上面的 pip 步骤（TE sdist 的依赖声明、requirements.txt）会顺带装进 nvidia-*-cu13
# 系列。致命之处在于 cu12 和 cu13 的包共用同一个安装路径
# （site-packages/nvidia/<lib>/lib/），后装的直接覆盖前者的 .so。
#
# 实测：nvidia-nccl-cu13 (2.30.7) 覆盖了 nvidia-nccl-cu12 (2.29.7) 的
# libnccl.so.2，而 CUDA 13 构建要求驱动 >= 580，于是 init_process_group 报
#   NCCL error ... NCCL version 2.30.7
#   Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'
#   (misc/cudawrap.cc initOnceFunc — NCCL 探测 driver API 版本时失败)
# 注意 torch.cuda.nccl.version() 报的是编译期头文件版本(2.29.7)，会误导；要用
#   strings site-packages/nvidia/nccl/lib/libnccl.so.2 | grep -oE '^2\.[0-9]+\.[0-9]+'
# 看文件真身。
#
# cudnn 同理，而且更危险：TE 的 fused attention 后端就靠 cuDNN，
# 也就是本配置实际使用的 attention 路径（见第 5 节 flash-attn 说明）。
CU13_PKGS=$(pip list 2>/dev/null | awk '/^nvidia-.*-cu13/{print $1}')
if [ -n "$CU13_PKGS" ]; then
  echo "[setup] 发现 cu13 包，替换为 cu12: $CU13_PKGS"
  pip uninstall -y $CU13_PKGS
  # 卸载会删掉共用路径下的 .so，所以必须 force-reinstall 把 cu12 版本写回去
  # （pip 认为 cu12 已装，不加 --force-reinstall 不会重写文件）
  pip install --force-reinstall --no-deps \
      "nvidia-nccl-cu12==2.29.7" "nvidia-cudnn-cu12==9.22.0.52" nvidia-cusparselt-cu12 nvidia-nvshmem-cu12
fi

# cudnn 版本必须 pin。这台机器上不 pin 会装到 9.20.0.48，然后
# torch.backends.cudnn.version() 抛
#   cuDNN version incompatibility: PyTorch was compiled against (9, 20, 0)
#   but found runtime version (9, 14, 0)   ← 或 (9, 5, 1)，取决于系统装了哪个
# 注意报错里的"运行时版本"不是这个包的版本：libcudnn.so.9 只有 ~130KB，是个
# dispatch shim，真正的实现在 libcudnn_graph / _ops / _cnn / _engines_*.so.9 里，
# 按 SONAME 在运行时解析。而系统的 /etc/ld.so.conf.d/ 往往把这些子库注册进
# ldconfig 缓存（本机指向 /usr/local/cuda-12.4/targets/x86_64-linux/lib 的 9.5.1），
# 其优先级独立于 LD_LIBRARY_PATH —— 所以"把 cuda-12.4 从 LD_LIBRARY_PATH 去掉"
# 并不能解决，实测清空后仍然复现。
# 9.22.0.52 是 Miles 官方 Dockerfile 的 cu12 分支所 pin 的版本，装上后 torch 报 92200。
pip install --force-reinstall --no-deps "nvidia-cudnn-cu12==9.22.0.52"

# 把系统 cudnn 从 ldconfig 缓存里摘掉 —— 这一步是必需的，不是优化。
#
# Miles 官方 Dockerfile 在同一处踩过并记录了成因（docker/Dockerfile 第 175 行附近）：
#   "The apt copy shadows the pip one in ldconfig and the loader interleaves the two,
#    so transformer_engine gets a mixed set of libcudnn sub-libraries."
# 它的解法是 apt-get remove --purge libcudnn9-cuda-12，也就是把系统那份删掉。
#
# 关键点：LD_LIBRARY_PATH 压不住它。ldconfig 缓存是**独立的解析路径**，加载器会把
# 两边交错取用，于是 libcudnn.so.9（9.22 的 shim）配上系统的 9.5.1 子库，TE 的
# FusedAttention 在 critic 第一次前向就报
#   cuDNN Error: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
#   cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# 实测：把 torch 自带 cudnn 目录前置到 LD_LIBRARY_PATH 并确认已传进 Ray worker，
# 报错一字不变；只有摘掉缓存条目才生效。
#
# 本机的系统 cudnn 来自 CUDA toolkit 而非独立 apt 包（/usr/local/cuda-12.4/targets/
# x86_64-linux/lib），所以注释掉注册它的 ld.so.conf.d 条目，而不是 apt remove。
# 这会让同目录的 cublas/cufft 等也退出缓存 —— 本栈安全：CUDA_HOME 指向 conda 的
# 12.9，torch/TE 用的是各自 pip 包里的库，靠 rpath 解析。pip 的 cudnn 本来就不进
# ldconfig 缓存（靠 rpath + LD_LIBRARY_PATH），所以缓存里为空才是正确状态。
for conf in /etc/ld.so.conf.d/*.conf; do
  [ -f "$conf" ] || continue
  # 只处理确实注册了 libcudnn 的目录，别误伤其他 conf（cublas/cufft 等照旧）
  has_cudnn=0
  while read -r d; do
    [ -n "$d" ] || continue
    case "$d" in \#*) continue ;; esac
    if [ -d "$d" ] && ls "$d"/libcudnn*.so* >/dev/null 2>&1; then has_cudnn=1; break; fi
  done < "$conf"
  [ "$has_cudnn" = 1 ] || continue
  echo "[setup] 从 ldconfig 摘掉系统 cudnn: $conf"
  cp -n "$conf" "$conf.justrl2.bak" 2>/dev/null || true
  sed -i 's|^\([^#].*\)$|#\1  # JustRL2: shadows pip cudnn, TE mixes sub-libraries|' "$conf"
done
ldconfig
if ldconfig -p | grep -q libcudnn; then
  echo "WARN: ldconfig 缓存里仍有 libcudnn，TE 可能混载子库："
  ldconfig -p | grep libcudnn | head -3
fi

# torch 自带 cudnn 目录仍然前置（缓存清空后由它提供子库）。
# 幂等：反复 activate 不重复叠加。
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], 'lib'))")
cat >> "$CONDA_PREFIX/etc/conda/activate.d/cuda129.sh" <<'EOF'
# torch 自带的 cudnn 子库目录。pip 的 cudnn 不进 ldconfig 缓存，需要显式在路径里。
# 另见上方把系统 cudnn 从 ldconfig 摘掉的那一步 —— 两者都要，缺一不可。
CUDNN_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib"
case ":$LD_LIBRARY_PATH:" in
  *":$CUDNN_LIB:"*) ;;
  *) export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
EOF
case ":${LD_LIBRARY_PATH:-}:" in
  *":$CUDNN_LIB:"*) ;;
  *) export LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

python -c "
import ctypes, torch
# shim 的版本号对不代表能用：实现都在子库里，按 SONAME 运行时解析。
ctypes.CDLL('libcudnn_graph.so.9')
v = torch.backends.cudnn.version()
print('cudnn', v, '(子库可加载) | nccl', torch.cuda.nccl.version())
assert v and v >= 92000, f'cudnn {v} 异常 —— 期望 >= 92000，见上方 ldconfig 说明'
"

# ---------------------------------------------------------------------------
# 8) numpy 收尾 —— 必须最后
# ---------------------------------------------------------------------------
# ray / miles / requirements.txt 都会把 numpy 升到 2.x，而 Megatron 在
# miles/backends/megatron_utils/initialize.py 里断言 NumPy 1.x。那个断言只在真正
# 启动训练时执行，所以现在不修，会等到 train.sh 跑起来才炸。
pip install httpx
pip install "numpy<2"

python -c "
import numpy, torch
assert numpy.__version__.startswith('1'), f'numpy {numpy.__version__} 必须是 1.x'
assert torch.__version__.startswith('2.13.0+cu129'), f'torch {torch.__version__} 被换掉了'
print('numpy', numpy.__version__, '| torch', torch.__version__)
"

echo "== 环境安装完成 =="
