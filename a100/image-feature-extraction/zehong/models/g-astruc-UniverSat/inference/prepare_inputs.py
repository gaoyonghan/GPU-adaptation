"""可选：把自有图片文件夹变成共享图片包。已有ImageNet图片包可直接复用。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image
from utils.protocol import sha256_file

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_inputs(source, output, count=None):
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
    parser.add_argument("--source", type=Path, required=True, help="原始图片文件夹，递归读取常见图片格式")
    parser.add_argument("--output", type=Path, required=True, help="尚不存在、位于source以外的新图片包目录")
    parser.add_argument("--count", type=int, help="可选固定选取数量；默认全部，不做类别均衡选图")
    args = parser.parse_args()
    print("Shared bundle:", prepare_inputs(args.source, args.output, args.count))


if __name__ == "__main__":
    main()
