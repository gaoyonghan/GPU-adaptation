"""把图片目录构造成跨设备共用的原始图片包。

新环境使用时，只需要把命令中的两个绝对路径改成当前服务器可访问的位置：

    python prepare_inputs.py \\
      --source /path/to/original_images \\
      --output /path/to/new_bundle

--source 是已有原始图片所在目录；--output 是新生成图片包的完整路径，必须
位于 source 目录之外且尚不存在。脚本不会覆盖已有目录。若两端已经有同一个
共享图片包，给 inference.py 传入 --bundle 指向该包路径，不必重复构造。

注意：可选的 --count 只按文件哈希排序后取前 N 张，不做类别均衡。它不能
重建本项目按 ImageNet 类别选出的 1000 张验收图片包。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image
from inference.protocol import sha256_file

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_inputs(source, output, count=None):
    """复制原始图片并生成评测器读取的清单，不做模型预处理。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output == source or output.is_relative_to(source):
        raise ValueError("Source must exist; output must be outside source tree")
    if output.exists():
        raise FileExistsError(f"Choose a new bundle path: {output}")
    candidates = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            with Image.open(path) as image:
                image.verify()
            candidates.append((sha256_file(path), path.relative_to(source).as_posix()))
    if not candidates or (count is not None and not 0 < count <= len(candidates)):
        raise ValueError("No images or invalid count")
    # 固定按内容哈希/路径排序选图；两端复制同一包，不分别挑选。
    candidates.sort()
    if count is not None:
        candidates = candidates[:count]
    (output / "images").mkdir(parents=True)
    with (output / "input_manifest.jsonl").open("x", encoding="utf-8") as handle:
        for digest, relative in candidates:
            sid = "img_" + hashlib.sha256((relative + "\n" + digest).encode()).hexdigest()[:32]
            target = f"images/{sid}.image"
            shutil.copyfile(source / relative, output / target)  # 保留原始编码字节。
            if sha256_file(output / target) != digest:
                raise ValueError(f"Source changed while copying: {relative}")
            handle.write(json.dumps({"sample_id": sid, "relative_path": target,
                                     "image_sha256": digest, "source_relative_path": relative}) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True,
        help="原始图片根目录（新环境需改成当前服务器上的实际路径）",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="新图片包绝对路径；目录必须不存在且在source目录之外",
    )
    parser.add_argument(
        "--count", type=int,
        help="可选图片数；按内容哈希排序取前N张，不做类别均衡，默认处理全部图片",
    )
    args = parser.parse_args()
    print("Shared bundle:", prepare_inputs(args.source, args.output, args.count))


if __name__ == "__main__":
    main()
