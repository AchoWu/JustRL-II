#!/usr/bin/env python3
"""体检裸机 cu129 环境 —— 逐条检查 bare_metal_cu129.sh 里记录过的真实故障点。

    python justrl2/setup/verify_env.py

只做 import 和版本检查，不占显存、不跑 kernel，几秒钟出结果。
配套 check_deps.py（列缺失的纯 Python 依赖）—— 两个都干净了再去跑 run_train.sh。

每条 FAIL 都对应一个已经踩过的坑，且大多数会在很晚才暴露（train.sh 跑起来、
init_process_group、甚至 save_checkpoint 时），所以在这里拦住是值得的。
"""

from __future__ import annotations

import ctypes
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
# 注意 TE 2.17 起顶层 transformer_engine 会把它 re-export 成
# transformer_engine.transformer_engine_torch，顶层模块名可能不存在 —— 两处都试。
_tet = None
for _name in ("transformer_engine_torch", "transformer_engine.transformer_engine_torch"):
    try:
        __import__(_name)
        _tet = _name
        break
    except ImportError as e:
        _tet_err = str(e)

if _tet:
    record(PASS, "transformer_engine_torch", f"import OK as {_tet} ({_ver('transformer-engine-torch')})")
elif "materialize_cow" in _tet_err:
    record(FAIL, "transformer_engine_torch", "ABI 断裂（materialize_cow_storage）—— 需按脚本第 4 节源码重编")
elif _ver("transformer-engine-torch"):
    record(WARN, "transformer_engine_torch", f"包已装({_ver('transformer-engine-torch')})但顶层模块不可 import；TE 自己能用即可")
else:
    record(FAIL, "transformer_engine_torch", f"未装 —— {_tet_err[:100]}")

try:
    import transformer_engine.pytorch as te_pt  # noqa: F401

    record(PASS, "transformer_engine.pytorch", "import OK")
except Exception as e:  # 不只是 ImportError：缺 onnxscript 会抛别的
    record(FAIL, "transformer_engine.pytorch", f"{type(e).__name__}: {str(e)[:120]}")

# cuDNN 是 TE fused attention 的后端，也就是本栈实际使用的 attention 路径。
# torch.backends.cudnn.version() 在运行时版本与编译期不匹配时**抛异常**而不是返回值，
# 报错文本里含 "cuDNN version incompatibility"。
#
# 有两个独立成因，修法完全不同，所以都要查：
#  (a) nvidia-cudnn-cu12 包版本比 torch 编译期用的旧 -> 升级包
#  (b) libcudnn.so.9 只是个 ~130KB 的 dispatch shim，真正的实现在
#      libcudnn_graph/_ops/_cnn/_engines_*.so.9 里，按 SONAME 在运行时解析。系统的
#      /etc/ld.so.conf.d/ 常把这些子库注册进 ldconfig 缓存，其优先级独立于
#      LD_LIBRARY_PATH —— 于是即使 LD_LIBRARY_PATH 干净、包版本也对，shim 仍会
#      加载系统那份旧子库。修法是把 torch 自带的 cudnn 目录放到 LD_LIBRARY_PATH 最前。
try:
    v = torch.backends.cudnn.version()
    if v and v >= 92000:
        record(PASS, "cuDNN", str(v))
    elif v and v >= 90000:
        record(WARN, "cuDNN", f"{v} —— 能用，但 torch 2.13 期望 9.20+；建议 nvidia-cudnn-cu12==9.22.0.52")
    else:
        record(FAIL, "cuDNN", f"{v} —— 异常，TE 的 FusedAttention 需要 cuDNN 9+")
except Exception as e:
    msg = str(e).replace("\n", " ")
    if "incompatibility" in msg:
        m = re.search(r"compiled\s+against\s+\((\d+),\s*(\d+),\s*(\d+)\).*?runtime version\s+\((\d+),\s*(\d+),\s*(\d+)\)", msg)
        detail = f"编译期 {'.'.join(m.groups()[:3])} vs 运行时 {'.'.join(m.groups()[3:])}" if m else msg[:90]
        record(FAIL, "cuDNN", f"版本不匹配（{detail}）")
        record(WARN, "  ^ nvidia-cudnn-cu12", f"{_ver('nvidia-cudnn-cu12') or '未装'}（建议 9.22.0.52）")
    else:
        record(FAIL, "cuDNN", f"查询失败: {msg[:110]}")

# 版本号对不代表能用：libcudnn.so.9 只是 ~130KB 的 dispatch shim，实现都在
# libcudnn_graph/_ops/_cnn/_engines_*.so.9 里，按 SONAME 运行时解析。
# 直接 dlopen 一个子库，这是唯一能提前发现混载的办法 —— 否则要等到 critic 第一次
# 前向才炸（TE FusedAttention: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED），而那已经是
# 二十多分钟之后。
try:
    ctypes.CDLL("libcudnn_graph.so.9")
    record(PASS, "cuDNN 子库", "libcudnn_graph.so.9 可加载")
except OSError as e:
    record(FAIL, "cuDNN 子库", f"libcudnn_graph.so.9 加载失败: {str(e)[:90]}")

# 「能载入一个」不等于「只载入了一套」。上面那条 CDLL 命中 LD_LIBRARY_PATH 最前面的
# 那份就 PASS，可致命的情形恰恰是**两份同时在进程里**：
#   .../site-packages/nvidia/cudnn/lib/libcudnn.so.9   (pip 9.22 的 dispatch shim)
#   $CONDA_PREFIX/lib/.../libcudnn.so.9.14.0           (conda install cudnn 带来的实现)
# 两者 SONAME 都是 libcudnn.so.9，先以 RTLD_GLOBAL 载入的赢得 cudnnBackend* 的符号
# 解析；9.14 的实现拿到调用后按 SONAME dlopen 自己那套子库，却从 LD_LIBRARY_PATH
# 拿到 9.22 的，于是 TE FusedAttention 在 critic 第一次前向报
#   cuDNN Error: ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# 而那已经是二十多分钟之后（实测一次 70 分钟）。
#
# 直接读 /proc/self/maps 看实际映射，这是唯一能区分「一套」和「两套」的办法：
# 版本号、CDLL、LD_LIBRARY_PATH 都看不出来。注意 maps 给的是 realpath，所以挪进
# 子目录（lib/_shadowed/）的那份照样会现原形 —— 实测挪子目录并不能让它退出视野。
# 依赖上面 section 4 已经 import 过 transformer_engine.pytorch（TE 是唯一的
# cuDNN 消费者），所以这里不重复 import。
try:
    _maps = Path("/proc/self/maps")
    if not _maps.exists():
        record(WARN, "cuDNN 单一来源", "/proc/self/maps 不可读，跳过")
    else:
        _libs = {
            Path(line.split()[-1])
            for line in _maps.read_text().splitlines()
            if "libcudnn" in line and line.split()[-1].startswith("/")
        }
        _dirs = sorted({str(p.parent) for p in _libs})
        if len(_dirs) > 1:
            record(FAIL, "cuDNN 单一来源", f"进程里有 {len(_dirs)} 个来源目录 —— 必然混载")
            for _d in _dirs:
                _names = sorted(p.name for p in _libs if str(p.parent) == _d)
                record(WARN, f"  ^ {_d}", ", ".join(_names))
            globals()["_cudnn_multi_dirs"] = _dirs
        elif _dirs:
            record(PASS, "cuDNN 单一来源", f"{len(_libs)} 个库全部来自 {_dirs[0]}")
        else:
            record(WARN, "cuDNN 单一来源", "进程里没有 libcudnn 映射 —— TE 是否真的 import 成功？")
except OSError as e:
    record(WARN, "cuDNN 单一来源", f"检查失败: {str(e)[:90]}")

# ldconfig 缓存里的 cudnn 是致命的：它是独立于 LD_LIBRARY_PATH 的解析路径，加载器会
# 把两边**交错**取用，于是 shim 是 pip 的 9.22、子库是系统的旧版。Miles 官方
# Dockerfile 为此显式 apt-get remove libcudnn9-cuda-12（见其注释 "the loader
# interleaves the two, so transformer_engine gets a mixed set"）。
# 实测把 torch 自带目录前置到 LD_LIBRARY_PATH（且确认已传进 Ray worker）无效，
# 只有摘掉缓存条目才行。
try:
    out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=20).stdout
    # 白名单是「pip 的 cudnn 那**一个**目录」，不是宽泛的 site-packages —— 后者会把
    # 别的 site-packages 里的第二份 cudnn 一起放过，而那和系统那份一样会混载。
    _pip_cudnn_dir = None
    try:
        import nvidia.cudnn

        _pip_cudnn_dir = str(Path(list(nvidia.cudnn.__path__)[0], "lib").resolve())
    except Exception:
        pass
    sysdirs = sorted(
        {
            d
            for line in out.splitlines()
            if "libcudnn" in line and "=>" in line
            for d in [str(Path(line.split("=>")[-1].strip()).resolve().parent)]
            if d != _pip_cudnn_dir
        }
    )
    if sysdirs:
        record(FAIL, "ldconfig cudnn", f"缓存里有系统 cudnn: {sysdirs} —— 会与 pip 的混载，见下方修复")
        globals()["_cudnn_ldconfig_dirs"] = sysdirs
    else:
        record(PASS, "ldconfig cudnn", "缓存内无系统 cudnn（正确 —— pip 的靠 rpath/LD_LIBRARY_PATH）")
except (FileNotFoundError, subprocess.TimeoutExpired):
    record(WARN, "ldconfig cudnn", "ldconfig 不可用，跳过")

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

    # nvidia.* 是 namespace package，__file__ 可能是 None；用 __path__ 找目录。
    roots = [Path(p) for p in getattr(nvidia.nccl, "__path__", [])]
    if nvidia.nccl.__file__:
        roots.insert(0, Path(nvidia.nccl.__file__).parent)
    sos = [so for root in roots for so in [root / "lib" / "libnccl.so.2"] if so.exists()]
    if sos:
        blob = sos[0].read_bytes()
        vers = sorted(set(re.findall(rb"2\.\d\d\.\d", blob)))
        real = b", ".join(vers).decode() if vers else "?"
        hdr = ".".join(str(x) for x in torch.cuda.nccl.version())
        if any(v.startswith(b"2.30") for v in vers):
            record(FAIL, "libnccl.so.2", f"文件里含 {real} —— 2.30.x 是 cu13 构建，要求 driver>=580")
        else:
            record(PASS, "libnccl.so.2", f"文件 {real} (torch 报头文件版本 {hdr})")
    else:
        record(WARN, "libnccl.so.2", f"在 {[str(r) for r in roots]} 下没找到")
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

# Triton 要 fork ptxas 去汇编 PTX。它按 TRITON_PTXAS_PATH 或 PATH 查找，而 Ray worker
# 的 PATH 不一定含 conda/CUDA 的 bin —— 那时 SGLang engine 会在 cuda graph capture 阶段
# 报 "RuntimeError: Cannot find ptxas" -> "Capture cuda graph failed"。
# train.sh 会自动探测并通过 runtime_env 传给 worker，这里只报告结果。
_ptxas = os.environ.get("TRITON_PTXAS_PATH", "")
if _ptxas and os.access(_ptxas, os.X_OK):
    record(PASS, "ptxas", f"TRITON_PTXAS_PATH={_ptxas}")
else:
    cands = []
    try:
        import triton

        cands.append(Path(Path(triton.__file__).parent, "backends", "nvidia", "bin", "ptxas"))
    except ImportError:
        pass
    cands.append(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "ptxas"))
    found = next((p for p in cands if p.exists() and os.access(p, os.X_OK)), None)
    if _ptxas:
        record(FAIL, "ptxas", f"TRITON_PTXAS_PATH={_ptxas} 不可执行")
    elif found:
        record(PASS, "ptxas", f"{found}（train.sh 会自动探测并传给 Ray worker）")
        _ptxas = str(found)
    else:
        record(FAIL, "ptxas", f"未找到，试过 {[str(p) for p in cands]} —— cuda graph capture 会失败")

# ptxas 必须在 LD_PRELOAD 注入的情况下也能起来。colocate 模式下 actor_group.py 会把
# torch_memory_saver 的 hook .so 通过 LD_PRELOAD 注入，Triton fork 出的 ptxas 继承它；
# 那个 hook 依赖 libcudart，若 cudart 不在 ldconfig 缓存里（例如为了处理 cudnn 而把整个
# ld.so.conf.d 条目注释掉了），ptxas 就起不来。而 NvidiaTool.from_path 只是吞掉
# CalledProcessError 返回 None，最终报成 "Cannot find ptxas" —— 完全指不到 cudart。
if _ptxas and os.access(_ptxas, os.X_OK):
    try:
        import torch_memory_saver

        hook = Path(
            Path(torch_memory_saver.__file__).parent.parent,
            "torch_memory_saver_hook_mode_preload.abi3.so",
        )
        if hook.exists():
            env = {**os.environ, "LD_PRELOAD": str(hook)}
            r = subprocess.run([_ptxas, "--version"], capture_output=True, text=True, timeout=30, env=env)
            if r.returncode == 0:
                record(PASS, "ptxas + LD_PRELOAD", "colocate 的 memory-saver hook 下可执行")
            else:
                miss = re.search(r"(lib[\w.]+\.so[\w.]*): cannot open", (r.stderr or "") + (r.stdout or ""))
                extra = f"缺 {miss.group(1)}" if miss else ((r.stderr or r.stdout or "")[:90]).strip()
                record(FAIL, "ptxas + LD_PRELOAD", f"起不来（{extra}）—— engine 会报 Cannot find ptxas")
        else:
            record(WARN, "ptxas + LD_PRELOAD", "找不到 memory-saver hook，跳过")
    except ImportError:
        record(WARN, "ptxas + LD_PRELOAD", "torch_memory_saver 未装，跳过")
    except subprocess.TimeoutExpired:
        record(WARN, "ptxas + LD_PRELOAD", "执行超时")

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
    # 标签带 "目录" 后缀，避免和下面的 import 检查同名 —— 修复建议是按标签匹配的，
    # 同名会让「框架目录缺失」在其实只是缺一个纯 Python 依赖时误触发。
    for d in ("Megatron-LM", "sglang"):
        p = repo / d
        if p.exists():
            record(PASS, f"{d} 目录", f"{p} -> {os.readlink(p) if p.is_symlink() else '(实体目录)'}")
        else:
            record(FAIL, f"{d} 目录", f"{p} 不存在 —— train.sh 会直接 exit 1")

    # train.sh 设 PYTHONPATH=.:Megatron-LM:sglang/python，这里手动补上以便 import 检查。
    for extra in (repo, repo / "Megatron-LM", repo / "sglang" / "python"):
        if extra.exists() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    for mod, label in (("megatron.core", "megatron"), ("sglang", "sglang import"), ("miles", "miles")):
        try:
            m = __import__(mod, fromlist=["__file__"])
            record(PASS, label, getattr(m, "__version__", "") or (getattr(m, "__file__", "") or "")[:70])
        except ModuleNotFoundError as e:
            # 缺一个纯 Python 依赖（sglang/miles 的依赖被 --no-deps 跳过了）和框架本身
            # 装坏了是两回事，前者 check_deps.py 一条命令就能补齐。
            missing = getattr(e, "name", "") or ""
            if missing and missing.split(".")[0] not in (mod.split(".")[0], "megatron", "sglang", "miles"):
                record(WARN, label, f"缺依赖 '{missing}' —— 跑 check_deps.py --pip 补齐（框架本身在位）")
            else:
                record(FAIL, label, f"ModuleNotFoundError: {str(e)[:110]}")
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

    failed = {name for lvl, name, _ in results if lvl == FAIL}
    fixes: list[str] = []

    if "ldconfig cudnn" in failed or "cuDNN 子库" in failed or "cuDNN" in failed:
        dirs = globals().get("_cudnn_ldconfig_dirs", [])
        lines = [
            "cuDNN 子库混载。libcudnn.so.9 只是 ~130KB 的 dispatch shim，实现在",
            "  libcudnn_graph/_ops/_cnn/_engines_*.so.9，按 SONAME 运行时解析。ldconfig 缓存是",
            "  独立于 LD_LIBRARY_PATH 的解析路径，加载器把两边交错取用 —— 于是 shim 用 pip 的",
            "  9.22、子库用系统的旧版，TE 的 FusedAttention 在 critic 首次前向报",
            "    cuDNN Error: ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED",
            "  cuDNN 是本栈唯一的 attention 后端（无 flash-attn），所以这是致命的。",
            "",
            "  Miles 官方 Dockerfile 为同一问题显式 apt-get remove libcudnn9-cuda-12，其注释：",
            '    "the loader interleaves the two, so transformer_engine gets a mixed set"',
            "  注意：把 torch 自带目录前置到 LD_LIBRARY_PATH **不管用**（已实测，即使确认",
            "  已传进 Ray worker，报错一字不变）。必须摘掉 ldconfig 缓存条目。",
        ]
        if dirs:
            lines += [
                "",
                f"  缓存里的系统 cudnn: {', '.join(dirs)}",
                "    # 只挪 libcudnn*，同目录的 cudart/cublas/cufft 必须留在原地",
                *[
                    f'    mkdir -p "{d}/_shadowed_cudnn" && mv "{d}"/libcudnn*.so* "{d}/_shadowed_cudnn/"'
                    for d in dirs
                ],
                "    ldconfig",
                "    ldconfig -p | grep libcudnn    # 应为空",
                "    ldconfig -p | grep libcudart   # 必须还在（见下）",
                "",
                "  切勿注释掉整个 ld.so.conf.d 条目图省事：那会连带把 libcudart.so.12 踢出缓存，",
                "  而 colocate 下 torch_memory_saver 的 hook 经 LD_PRELOAD 注入且依赖 cudart，",
                "  Triton fork 的 ptxas 继承它后起不来，最终报成 'Cannot find ptxas'，指不到 cudart。",
                "",
                "  之后必须重启 Ray，worker 才会读到新缓存：",
                "    ray stop --force && bash run_train.sh",
            ]
        fixes.append("\n".join(lines))

    if "cuDNN 单一来源" in failed:
        dirs = globals().get("_cudnn_multi_dirs", [])
        lines = [
            "进程里同时载入了两套 cuDNN。两份的 SONAME 都是 libcudnn.so.9，先以 RTLD_GLOBAL",
            "  载入的那份赢得 cudnnBackend* 的符号解析；它再按 SONAME dlopen 自己那套子库时",
            "  却拿到另一份的版本，于是 TE FusedAttention 在 critic 首次前向报",
            "    cuDNN Error: ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED",
            "",
            "  最常见来源是 bare_metal_cu129.sh 里的 `conda install -c nvidia cudnn`（为了拿",
            "  cudnn.h 给 TE 的 torch 扩展编译）—— 它同时把一整套运行时库装进 $CONDA_PREFIX/lib。",
            "  这和脚本里 nccl 那段警告是同一个机制，只是 cudnn 更致命（唯一的 attention 后端）。",
            "",
            "  修法：头文件留下，运行时库移出 conda prefix。注意必须移到 prefix **之外** ——",
            "  实测挪到 $CONDA_PREFIX/lib 的子目录（如 lib/_shadowed/）仍会被载入。",
            "    mkdir -p /root/cudnn_conda_backup",
            '    mv "$CONDA_PREFIX"/lib/libcudnn*.so* /root/cudnn_conda_backup/',
            "    # 复查：下面应只打出 nvidia/cudnn/lib 一个目录",
            "    python -c \"import torch, transformer_engine.pytorch;\"\\",
            "\"print(sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libcudnn' in l}))\"",
        ]
        if dirs:
            lines += ["", f"  本次检测到的来源目录: {', '.join(dirs)}"]
        lines += [
            "",
            "  之后必须重启 Ray，worker 才会用新的加载结果：",
            "    ray stop --force && bash run_train.sh",
        ]
        fixes.append("\n".join(lines))

    if "Megatron-LM 目录" in failed or "sglang 目录" in failed:
        fixes.append(
            "框架目录缺失（train.sh 会直接 exit 1）。必须在仓库根目录执行：\n"
            "    git clone --depth 1 -b miles-main https://github.com/radixark/Megatron-LM.git _Megatron-LM \\\n"
            "      && ln -sfn _Megatron-LM Megatron-LM\n"
            "    git clone --depth 1 -b sglang-miles https://github.com/sgl-project/sglang.git _sglang \\\n"
            "      && ln -sfn _sglang sglang"
        )

    if "cu13 包" in failed or "libnccl.so.2" in failed:
        fixes.append(
            "cu13 包污染（cu12/cu13 共用 site-packages/nvidia/<lib>/lib/，后装的覆盖 .so）：\n"
            "    pip uninstall -y $(pip list 2>/dev/null | awk '/^nvidia-.*-cu13/{print $1}')\n"
            "    pip install --force-reinstall --no-deps 'nvidia-nccl-cu12==2.29.7' \\\n"
            "        nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nvshmem-cu12\n"
            "  （--force-reinstall 是必需的：卸载删掉了共用路径下的 .so，pip 认为 cu12 已装不会重写）"
        )

    if "flash-attn" in failed:
        fixes.append("pip uninstall -y flash-attn   # 装着它 TE 就会去 import 然后崩")

    if "numpy" in failed:
        fixes.append("pip install 'numpy<2'   # Megatron 硬断言 1.x")

    if "nvcc" in failed:
        fixes.append(
            "nvcc 不是 12.9（JIT 会报 Unknown option '--compress-mode=size'）：\n"
            "    conda deactivate && conda activate " + (os.environ.get("CONDA_DEFAULT_ENV") or "justrl2") + "\n"
            "  activate.d/cuda129.sh 会设 CUDA_HOME/PATH；不生效就检查那个文件是否存在。"
        )

    if fixes:
        print("\n--- 修复建议 ---")
        for i, f in enumerate(fixes, 1):
            print(f"\n{i}. {f}")

    print("\n修完再跑一遍本脚本，然后 python justrl2/setup/check_deps.py --pip 补纯 Python 依赖。")
    sys.exit(1)

print("\n没有 FAIL。下一步：")
print("  python justrl2/setup/check_deps.py --pip   # 补齐纯 Python 依赖")
print("  bash run_train.sh                          # 3-rollout 冒烟，自带启动检查")
sys.exit(0)
