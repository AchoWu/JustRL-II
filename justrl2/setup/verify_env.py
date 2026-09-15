#!/usr/bin/env python3
"""体检裸机 cu129 环境 —— 逐条检查 bare_metal_cu129.sh 里记录过的真实故障点。

    python justrl2/setup/verify_env.py

只做 import 和版本检查，不占显存、不跑 kernel，几秒钟出结果。
配套 check_deps.py（列缺失的纯 Python 依赖）—— 两个都干净了再去跑 run_train.sh。

每条 FAIL 都对应一个已经踩过的坑，且大多数会在很晚才暴露（train.sh 跑起来、
init_process_group、甚至 save_checkpoint 时），所以在这里拦住是值得的。
"""

from __future__ import annotations

import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str) -> None:
    results.append((level, name, detail))
    color = {"PASS": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m"}[level]
    print(f"  {color}{level}\033[0m  {name}: {detail}", flush=True)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
section("1) torch / CUDA / GPU")
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as e:
    record(FAIL, "torch", f"import 失败: {e}")
    print("\ntorch 都没有，后面免谈。检查是否 activate 了正确的 conda 环境。")
    sys.exit(1)

# 必须带 +cu129 本地版本号。PEP 440 下 2.13.0+cu130 也满足 ==2.13.0，
# 不带本地版本号的约束形同虚设，很容易被后续 pip 悄悄换成 cu130 构建。
if torch.__version__.startswith("2.13.0+cu129"):
    record(PASS, "torch", torch.__version__)
else:
    record(FAIL, "torch", f"{torch.__version__} —— 期望 2.13.0+cu129，被别的 pip 步骤换掉了")

if torch.version.cuda and torch.version.cuda.startswith("12"):
    record(PASS, "torch.version.cuda", torch.version.cuda)
else:
    record(FAIL, "torch.version.cuda", f"{torch.version.cuda} —— driver 535 用不了 CUDA 13 构建")

# torch.cuda.is_available() 只做轻量探测，不建 context，安全。
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    record(PASS, "GPU", f"{n} x {name}, sm{cap[0]}{cap[1]}")
    if cap != (9, 0):
        record(WARN, "GPU arch", f"sm{cap[0]}{cap[1]} —— 本栈只按 sm90 (H100/H20) 验证过")
else:
    record(FAIL, "torch.cuda.is_available()", "False —— 驱动/容器 GPU 透传有问题，或装成了 cu13 包")

# driver 535 最高 JIT 到 PTX ISA 8.2，CUDA 12.9 的 nvcc 生成 8.8。
# 带 +PTX 的 arch list 会让源码编译产出只能靠 JIT 的 PTX，运行时报 CUDA error 222。
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
if not arch_list:
    record(WARN, "TORCH_CUDA_ARCH_LIST", "未设置 —— 只影响源码编译；若还要编东西请设成 9.0")
elif "PTX" in arch_list.upper():
    record(FAIL, "TORCH_CUDA_ARCH_LIST", f"{arch_list} —— 绝不能带 +PTX，driver 535 只到 PTX 8.2")
else:
    record(PASS, "TORCH_CUDA_ARCH_LIST", arch_list)

# ---------------------------------------------------------------------------
section("2) numpy —— Megatron 断言 1.x")
# ---------------------------------------------------------------------------
# miles/backends/megatron_utils/initialize.py 里的断言只在真正启动训练时执行，
# 所以这里不查会等到 train.sh 跑起来才炸。
try:
    import numpy

    if numpy.__version__.startswith("1."):
        record(PASS, "numpy", numpy.__version__)
    else:
        record(FAIL, "numpy", f"{numpy.__version__} —— Megatron 断言 1.x，跑 `pip install 'numpy<2'`")
except ImportError as e:
    record(FAIL, "numpy", f"import 失败: {e}")

# ---------------------------------------------------------------------------
section("3) flash-attn —— 本栈必须缺席")
# ---------------------------------------------------------------------------
# torch 2.13 移除了 c10::impl::cow::materialize_cow_storage，torch 2.2~2.12 构建的
# 每个 flash-attn wheel（含官方 Dao-AILab）都引用它，import 就炸。而 TE 2.17 的
# backends.py 是按包 metadata 判断的：只要 importlib.metadata 能查到 flash-attn，
# 它就会去 import flash_attn_2_cuda 然后崩 —— NVTE_FLASH_ATTN=0 救不了。
fa = _ver("flash-attn")
if fa is None:
    record(PASS, "flash-attn", "未安装（正确 —— TE 会走 cuDNN FusedAttention）")
else:
    record(FAIL, "flash-attn", f"装了 {fa} —— 必须 `pip uninstall -y flash-attn`，否则 TE 直接挂")

for extra in ("flash-attn-3", "ring-flash-attn"):
    v = _ver(extra)
    if v:
        record(WARN, extra, f"装了 {v} —— 内部 import flash_attn，本栈用不到")

# ---------------------------------------------------------------------------
section("4) TransformerEngine —— 实际的 attention 后端")
# ---------------------------------------------------------------------------
te_core = _ver("transformer-engine-cu12")
if te_core:
    record(PASS, "transformer_engine_cu12", te_core)
elif _ver("transformer-engine-cu13"):
    record(FAIL, "transformer_engine", "装的是 cu13 核心 —— driver 535 用不了")
else:
    record(WARN, "transformer_engine_cu12", "查不到版本")

# transformer_engine_torch 是 ABI 断裂的重灾区：预编译 wheel 针对 torch <= 2.12，
# 在 2.13 上 import 报 undefined symbol: ..._ZN3c104impl3cow23materialize_cow_storage...
try:
    import transformer_engine_torch  # noqa: F401

    record(PASS, "transformer_engine_torch", f"import OK ({_ver('transformer-engine-torch')})")
except ImportError as e:
    msg = str(e)
    if "materialize_cow" in msg:
        record(FAIL, "transformer_engine_torch", "ABI 断裂（materialize_cow_storage）—— 需按脚本第 4 节源码重编")
    else:
        record(FAIL, "transformer_engine_torch", f"import 失败: {msg[:120]}")

try:
    import transformer_engine.pytorch as te_pt  # noqa: F401

    record(PASS, "transformer_engine.pytorch", "import OK")
except Exception as e:  # 不只是 ImportError：缺 onnxscript 会抛别的
    record(FAIL, "transformer_engine.pytorch", f"{type(e).__name__}: {str(e)[:120]}")

# cuDNN 是 TE fused attention 的后端，也就是本栈实际使用的 attention 路径。
try:
    v = torch.backends.cudnn.version()
    if v and v >= 90000:
        record(PASS, "cuDNN", str(v))
    else:
        record(FAIL, "cuDNN", f"{v} —— 异常，TE 的 FusedAttention 需要 cuDNN 9+")
except Exception as e:
    record(FAIL, "cuDNN", f"查询失败: {e}")

# ---------------------------------------------------------------------------
section("5) cu13 包污染 —— cu12/cu13 共用安装路径，后装的覆盖前者 .so")
# ---------------------------------------------------------------------------
cu13 = sorted(
    d.metadata["Name"]
    for d in md.distributions()
    if d.metadata.get("Name", "").startswith("nvidia-") and d.metadata["Name"].endswith("-cu13")
)
if cu13:
    record(FAIL, "cu13 包", f"{' '.join(cu13)} —— 见脚本第 7 节，卸掉后必须 --force-reinstall cu12 版本")
else:
    record(PASS, "cu13 包", "无")

# torch.cuda.nccl.version() 报的是编译期头文件版本，会误导；要看 .so 文件真身。
# cu13 的 NCCL 2.30.7 要求 driver >= 580，会在 init_process_group 时才报
# "Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'"。
try:
    import nvidia.nccl

    so = Path(nvidia.nccl.__file__).parent / "lib" / "libnccl.so.2"
    if so.exists():
        blob = so.read_bytes()
        vers = sorted(set(re.findall(rb"2\.\d\d\.\d", blob)))
        real = b", ".join(vers).decode() if vers else "?"
        hdr = ".".join(str(x) for x in torch.cuda.nccl.version())
        if any(v.startswith(b"2.30") for v in vers):
            record(FAIL, "libnccl.so.2", f"文件里含 {real} —— 2.30.x 是 cu13 构建，要求 driver>=580")
        else:
            record(PASS, "libnccl.so.2", f"文件 {real} (torch 报头文件版本 {hdr})")
    else:
        record(WARN, "libnccl.so.2", f"没找到 {so}")
except ImportError:
    record(WARN, "nvidia.nccl", "查不到 —— torch 可能链接的是系统 NCCL")

# conda 的 libnccl 会在 dlopen 时抢先加载，遮蔽 torch 自带的那个。
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    stray = list(Path(conda_prefix, "lib").glob("libnccl.so*"))
    if stray:
        record(FAIL, "conda lib/libnccl.so*", f"{[p.name for p in stray]} —— 会遮蔽 torch 的，挪出 lib/")
    else:
        record(PASS, "conda lib/libnccl.so*", "无（正确）")

# ---------------------------------------------------------------------------
section("6) rollout 侧 kernel")
# ---------------------------------------------------------------------------
for pkg, mod in (("sglang-kernel", "sgl_kernel"), ("flashinfer-python", "flashinfer")):
    v = _ver(pkg) or _ver(pkg.replace("sglang-", "sgl-"))
    try:
        __import__(mod)
        record(PASS, mod, f"import OK ({v})")
    except Exception as e:
        record(FAIL, mod, f"{type(e).__name__}: {str(e)[:110]}")

# jit-cache 提供预编译 sm90 cubin。缺了 flashinfer 会在首次调用时现场编译 ——
# 慢，且 8 个 rollout worker 会互相打架。
if _ver("flashinfer-jit-cache"):
    record(PASS, "flashinfer-jit-cache", _ver("flashinfer-jit-cache"))
else:
    record(WARN, "flashinfer-jit-cache", "未装 —— flashinfer 会运行时 JIT，慢且多 worker 打架")

# ---------------------------------------------------------------------------
section("7) apex fused wgrad —— 决定要不要 NO_GRAD_ACC_FUSION=1")
# ---------------------------------------------------------------------------
# 顶层 apex 是纯 Python，import 没问题；CUDA 扩展有同样的 torch 2.13 ABI 断裂，
# 只在 ColumnParallelLinear 真正用到时才暴露。
try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401

    record(PASS, "fused_weight_gradient_mlp_cuda", "可用 —— 不需要 NO_GRAD_ACC_FUSION")
except ImportError:
    record(
        WARN,
        "fused_weight_gradient_mlp_cuda",
        "不可用 —— 必须用 -baremetal.env（含 NO_GRAD_ACC_FUSION=1），否则训练时报错",
    )

# ---------------------------------------------------------------------------
section("8) 框架就位 —— train.sh 硬检查这两个目录")
# ---------------------------------------------------------------------------
repo = None
for base in (Path.cwd(), *Path(__file__).resolve().parents):
    if (base / "requirements.txt").exists() and (base / "justrl2").is_dir():
        repo = base
        break

if repo is None:
    record(FAIL, "仓库根目录", "找不到（需同时看到 requirements.txt 和 justrl2/）")
else:
    record(PASS, "仓库根目录", str(repo))
    for d in ("Megatron-LM", "sglang"):
        p = repo / d
        if p.exists():
            record(PASS, d, f"{p} -> {os.readlink(p) if p.is_symlink() else '(实体目录)'}")
        else:
            record(FAIL, d, f"{p} 不存在 —— train.sh 会直接 exit 1")

    # train.sh 设 PYTHONPATH=.:Megatron-LM:sglang/python，这里手动补上以便 import 检查。
    for extra in (repo, repo / "Megatron-LM", repo / "sglang" / "python"):
        if extra.exists() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    for mod, label in (("megatron.core", "megatron"), ("sglang", "sglang"), ("miles", "miles")):
        try:
            m = __import__(mod, fromlist=["__file__"])
            record(PASS, label, getattr(m, "__version__", "") or (getattr(m, "__file__", "") or "")[:70])
        except Exception as e:
            record(FAIL, label, f"{type(e).__name__}: {str(e)[:110]}")

# ---------------------------------------------------------------------------
section("9) 数据/权重转换链路")
# ---------------------------------------------------------------------------
for pkg, mod, why in (
    ("mbridge", "mbridge", "HF->Megatron 权重映射，prepare_model.sh 用"),
    ("math-verify", "math_verify", "--rm-type math 的答案判定"),
    ("safetensors", "safetensors", "prepare_model.sh 的 bin->safetensors 转换"),
    ("psutil", "psutil", "Megatron 异步 dist-ckpt writer，缺了在 save 时才报"),
    ("ray", "ray", "整个训练的调度层"),
):
    try:
        __import__(mod)
        record(PASS, mod, _ver(pkg) or "已装")
    except ImportError:
        record(FAIL, mod, f"未装 —— {why}")

# hf download 是 prepare_model.sh 用的新版 CLI（不是旧的 huggingface-cli）
try:
    out = subprocess.run(["hf", "--version"], capture_output=True, text=True, timeout=20)
    if out.returncode == 0:
        record(PASS, "hf CLI", out.stdout.strip()[:60] or "OK")
    else:
        record(FAIL, "hf CLI", "调用失败 —— pip install -U 'huggingface_hub[cli,hf_transfer]'")
except (FileNotFoundError, subprocess.TimeoutExpired):
    record(FAIL, "hf CLI", "没找到 —— prepare_model.sh 会报 'hf: command not found'")

# ---------------------------------------------------------------------------
section("10) nvcc —— 任何 JIT / 源码编译都要它是 12.9")
# ---------------------------------------------------------------------------
# flashinfer / TE / Megatron 的 JIT 会传 --compress-mode=size（nvcc >= 12.8 才有），
# 系统的 12.4 会报 "nvcc fatal: Unknown option '--compress-mode=size'"。
try:
    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=20)
    m = re.search(r"release (\d+\.\d+)", out.stdout)
    ver = m.group(1) if m else "?"
    if ver == "12.9":
        record(PASS, "nvcc", f"{ver} ({os.environ.get('CUDA_HOME', 'CUDA_HOME 未设')})")
    else:
        record(
            FAIL,
            "nvcc",
            f"{ver} —— 期望 12.9；CUDA_HOME/PATH 是否丢了？重新 activate 环境（见脚本第 2 节）",
        )
except (FileNotFoundError, subprocess.TimeoutExpired):
    record(WARN, "nvcc", "没找到 —— 只在需要 JIT / 源码编译时才是问题")

# ---------------------------------------------------------------------------
print()
n_fail = sum(1 for lvl, _, _ in results if lvl == FAIL)
n_warn = sum(1 for lvl, _, _ in results if lvl == WARN)
print("=" * 68)
print(f"  {len(results)} 项检查：{len(results) - n_fail - n_warn} PASS / {n_warn} WARN / {n_fail} FAIL")
print("=" * 68)

if n_fail:
    print("\nFAIL 项：")
    for lvl, name, detail in results:
        if lvl == FAIL:
            print(f"  - {name}: {detail}")
    print("\n先修掉这些，再跑 python justrl2/setup/check_deps.py --pip 补纯 Python 依赖。")
    sys.exit(1)

print("\n没有 FAIL。下一步：")
print("  python justrl2/setup/check_deps.py --pip   # 补齐纯 Python 依赖")
print("  bash run_train.sh                          # 3-rollout 冒烟，自带启动检查")
sys.exit(0)
