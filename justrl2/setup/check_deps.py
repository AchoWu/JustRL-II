#!/usr/bin/env python3
"""列出 sglang 和 miles 声明了但环境里没装的依赖。

    python justrl2/setup/check_deps.py            # 只报告
    python justrl2/setup/check_deps.py --pip      # 额外打印可直接执行的 pip 命令

裸机路径上 sglang 只被 clone 到 PYTHONPATH、miles 用 `pip install -e . --no-deps`
装的（都是为了防止 pip 拖进 cu13 包覆盖已验证的 cu129 栈），代价是纯 Python 依赖
一并被跳过，然后在 import 时一个一个冒出来。这个脚本一次列全。
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib

from packaging.requirements import Requirement

# 从 cwd 或脚本位置向上找仓库根，这样从哪跑、复制到哪都能用。
REPO = None
for _base in (Path.cwd(), *Path(__file__).resolve().parents):
    if (_base / "requirements.txt").exists() and (_base / "justrl2").is_dir():
        REPO = _base
        break
if REPO is None:
    raise SystemExit("找不到仓库根目录（需要能同时看到 requirements.txt 和 justrl2/）")

# 这几个由 bare_metal_cu129.sh 手工管理（特定 wheel / 本地版本号 +cu129 / 必须缺席），
# 不能让 pip 按声明去解析：PyPI 上的 sglang-kernel 是 CUDA 13 构建，装上会重现
# "CUDA driver version is insufficient"；flash-attn 装回来 TE 立刻挂。
MANUAL = {
    "torch", "torchvision", "torchaudio", "numpy",
    "sglang-kernel", "sgl-kernel", "flashinfer-python", "flashinfer-jit-cache",
    "transformer-engine", "transformer-engine-cu12", "transformer-engine-torch",
    "apex", "sglang", "sglang-router",
    "nvidia-nccl-cu12", "nvidia-cudnn-cu12",
}

# 这些在本栈上不装，各有原因。dense Llama + Megatron + 单机 colocate 都用不到。
EXCLUDE = {
    # torch 2.13 移除了 c10::impl::cow::materialize_cow_storage，torch 2.2~2.12 构建的
    # flash-attn wheel 全部 import 就炸；且 TE 只要检测到包存在就会去 import。
    "flash-attn": "ABI 断裂，装上 TE 直接挂",
    "flash-attn-3": "同上",
    "flash-attn-4": "同上",
    "ring-flash-attn": "内部 import flash_attn；只有 FSDP 路径用，Megatron 不需要",
    # MoE expert-parallel 专用，且 JIT 重度、cu129 无预编译 wheel
    "sgl-deep-ep": "MoE EP 专用，JIT 重度，本模型是 dense Llama",
    "sgl-deep-gemm": "同上",
    "deep-ep": "同上",
    # 会拖 torch 或需要特定 CUDA 构建
    "torchft-nightly": "容错训练，不需要，且可能覆盖 torch",
    "torchcodec": "音视频解码，VLM 才用，版本与 torch 强绑定",
    "nvidia-mathdx": "CUDA 13 侧构建",
}


def _missing(deps: list[str]) -> tuple[list[str], list[str], list[str]]:
    missing, manual, excluded = [], [], []
    for raw in deps:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.marker and not req.marker.evaluate():
            continue
        try:
            md.version(req.name)
            continue  # 已装
        except md.PackageNotFoundError:
            pass
        name = req.name.lower().replace("_", "-")
        if name in MANUAL:
            manual.append(req.name)
        elif name in EXCLUDE:
            excluded.append(name)
        else:
            missing.append(req.name)
    return missing, manual, excluded


def _report(label: str, deps: list[str]) -> list[str]:
    missing, manual, excluded = _missing(deps)
    print(f"\n=== {label}: {len(deps)} 个依赖，待装 {len(missing)} 个 ===")
    print("  " + (" ".join(sorted(missing)) if missing else "(无)"))
    if manual:
        print(f"  [手工管理，不由 pip 装] {' '.join(sorted(manual))}")
    for name in sorted(excluded):
        print(f"  [跳过] {name} — {EXCLUDE[name]}")
    return missing


def main() -> None:
    print(f"repo: {REPO}")
    all_missing: set[str] = set()

    pyproject = REPO / "sglang" / "python" / "pyproject.toml"
    if pyproject.exists():
        deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
        all_missing |= set(_report("sglang", deps))
    else:
        print(f"!! 没找到 {pyproject}，sglang 是否已 clone？")

    lines = [ln.split("#")[0].strip() for ln in (REPO / "requirements.txt").read_text().splitlines()]
    all_missing |= set(_report("miles requirements.txt", [ln for ln in lines if ln]))

    print(f"\n=== 合并去重：{len(all_missing)} 个 ===")
    print("  " + (" ".join(sorted(all_missing)) if all_missing else "(无)"))

    if all_missing and "--pip" in sys.argv:
        # 直接把 constraints 写到盘上，避免复制 heredoc 时漏掉那几行。
        cons = Path("/tmp/justrl2_constraints.txt")
        cons.write_text(
            "torch==2.13.0\ntorchvision==0.28.0\ntorchaudio==2.11.0\n"
            "numpy<2\nsglang-router<0.3.2\n"
        )
        print(f"\nconstraints 已写入 {cons}")
        print("\n--- 复制执行 ---")
        print(f"pip install -c {cons} {' '.join(sorted(all_missing))}")
        print("\n--- 装完必查（torch 带 +cu129 / numpy 1.x / flash-attn 那行必须为空）---")
        print("pip list 2>/dev/null | grep -iE '^(torch|numpy|sglang-kernel|flash-attn) '")
        print("warning: should pip nvidia-cudnn-cu12==9.22.0.52")
        print('pip install --force-reinstall --no-deps "nvidia-cudnn-cu12==9.22.0.52"')


if __name__ == "__main__":
    main()
