#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单模型推理的原生入口，由 ../../inference.py 按 entrypoint.json 调起。

职责边界：
  - 解析路径（权重、数据集、输出目录），路径不写死在代码里；
  - 数据集缺失时按 model.json 的 prepare 声明补齐（准备步骤属于推理入口的一部分）；
  - 调 engine.run() 跑推理并落盘 predictions.jsonl / status.json / timing.json。

统一评测需要的格式转换不在这里做——那一步由 ../prepare.py 完成。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <model>/inference/native
MODEL_ROOT = HERE.parents[1]                    # <model>
CONFIG = json.loads((HERE / "model.json").read_text(encoding="utf-8"))

# 权重默认随交付目录落盘在 <model>/weights/；缺失时才回退到 MX 共享目录。
DEFAULT_WEIGHTS_ROOT = "/mnt/afs/laishen/gpu_infer/models"


def resolve_dataset_root(cli_value=None) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("DATASET_ROOT")
    if env:
        return Path(env).resolve()
    # <project>/classification/<contributor>/models/<model> -> <project>/dataset
    return MODEL_ROOT.parents[3].resolve() / "dataset"


def resolve_weights_root(cli_value=None) -> Path:
    """权重根目录：命令行 > 环境变量 > 交付目录内的 weights/ > MX 共享目录。"""
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("WEIGHTS_ROOT")
    if env:
        return Path(env).resolve()
    local = MODEL_ROOT / "weights"
    if (local / "config.json").is_file():
        return local
    return Path(DEFAULT_WEIGHTS_ROOT)


def resolve_weights_dir(weights_root: Path) -> Path:
    """实际加载目录：优先 <root>/<dirname>；交付目录里权重就在 weights/ 这一层。"""
    target = weights_root / CONFIG["dirname"]
    if target.is_dir():
        return target
    if (weights_root / "config.json").is_file() and any(
            (weights_root / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
        return weights_root
    return target


def dataset_ready(path: Path) -> bool:
    """评测集是否已就绪：parquet 目录，或 ImageFolder 的类目录布局，二者任一。"""
    if not path.is_dir():
        return False
    if any(path.glob("*.parquet")) or any(path.rglob("*.parquet")):
        return True
    if any((path / n).is_dir() for n in ("val", "test", "validation")):
        return True
    return any(d.is_dir() for d in path.iterdir())


def ensure_dataset(dataset_root: Path) -> None:
    """评测集不在共享 dataset/ 时，按 model.json 的 prepare 声明现场准备。"""
    target = dataset_root / CONFIG["dataset_sub"]
    if dataset_ready(target):
        return

    prep = CONFIG.get("prepare")
    if not prep:
        raise SystemExit(
            f"评测集缺失：{target}\n"
            f"该模型没有自动准备步骤，请把数据集放到共享 dataset/{CONFIG['dataset_sub']}，"
            f"或用 --dataset-root 指向已有目录。")

    script = HERE / "prepare_dataset.py"
    if not script.is_file():
        raise SystemExit(f"评测集缺失：{target}，且找不到准备脚本 {script}")

    cmd = [sys.executable, str(script), *prep.get("args", [])]
    env = dict(os.environ)
    env["DATA_DIR"] = str(dataset_root)
    env["MODEL_DIR"] = str(resolve_weights_root())
    print(f"  评测集缺失，先执行准备步骤：{' '.join(cmd)}")
    print(f"  DATA_DIR={env['DATA_DIR']}  MODEL_DIR={env['MODEL_DIR']}")
    subprocess.run(cmd, env=env, check=True)

    if not dataset_ready(target):
        raise SystemExit(f"准备步骤执行完仍没有 {target}，请检查 prepare_dataset.py 的输出")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--phase", choices=["full", "smoke", "preflight"], default="full")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="本次运行的输出目录（默认 <model>/result/run_001）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10, help="预热批次数，不计入计时")
    ap.add_argument("--max-images", type=int, default=0, help="冒烟用：最多评测多少张")
    ap.add_argument("--precision", choices=["amp_bf16", "amp_fp16", "fp32"], default="amp_bf16")
    ap.add_argument("--dataset-root", default=None, help="覆盖共享数据集根目录")
    ap.add_argument("--weights-root", default=None, help="覆盖权重根目录")
    args = ap.parse_args()

    out_root = (args.output_root or (MODEL_ROOT / "result" / "run_001")).resolve()
    phase = args.phase
    max_images = args.max_images or (10 if phase == "smoke" else None)

    dataset_root = resolve_dataset_root(args.dataset_root)
    weights_root = resolve_weights_root(args.weights_root)

    if phase == "preflight":
        print(json.dumps({
            "sequence": CONFIG["sequence"], "model_id": CONFIG["model_id"],
            "model_root": str(MODEL_ROOT), "weights": str(resolve_weights_dir(weights_root)),
            "dataset": str(dataset_root / CONFIG["dataset_sub"]),
            "output_root": str(out_root), "python": sys.executable,
        }, ensure_ascii=False, indent=2))
        return 0

    ensure_dataset(dataset_root)

    # 只有冒烟才需要打破「张数必须等于登记值」的约束，预热也相应取消
    warmup = 0 if max_images else args.warmup

    sys.path.insert(0, str(HERE))
    import engine                                    # noqa: E402

    engine.run(CONFIG, MODEL_ROOT, weights_root, dataset_root, out_root,
               device_str=args.device, batch_size=args.batch_size, workers=args.workers,
               warmup=warmup, precision=args.precision, max_images=max_images, phase=phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
