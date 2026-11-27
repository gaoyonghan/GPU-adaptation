#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet 读写小工具 —— 统一「评测集」的落盘格式

为什么要有这个文件：
  推理引擎 run_inference.py 只认 parquet（read_shard 读 image + label/labels 两列）。
  而 167/168/170 的原始素材形态各异：
    167  HF 数据集（parquet，但 11 个 style 要筛 3 个、标签标签要换到模型索引空间）
    168  SD2.1 现场生成的图片（内存里的 PIL 对象）
    170  13.28GB 的 zip（解压出文件夹结构）
  与其让引擎长出一堆 loader，不如在这里把异构脏活收敛掉：
  一切都在 preprocess.py 里转成「image=binary + label=int64」的 parquet，引擎零改动。

schema 刻意用 binary（不是 HF 的 struct<bytes,path>）：
  run_inference._image_bytes() 对 bytes/bytearray/memoryview 与 dict 两种都支持，
  但 binary 更省一层解析，落盘也小。
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("image", pa.binary()),
    pa.field("label", pa.int64()),
])


class ParquetWriter:
    """分块写 parquet，避免把上万张图一次性堆在内存里。"""

    def __init__(self, path: str | Path, batch_rows: int = 500, compression: str = "zstd"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._w = pq.ParquetWriter(str(self.path), SCHEMA, compression=compression)
        self._buf: list[tuple[bytes, int]] = []
        self._batch = batch_rows
        self.n = 0

    def add(self, img_bytes: bytes, label: int) -> None:
        self._buf.append((bytes(img_bytes), int(label)))
        if len(self._buf) >= self._batch:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        imgs = [b for b, _ in self._buf]
        labs = [y for _, y in self._buf]
        t = pa.table(
            {"image": pa.array(imgs, type=pa.binary()),
             "label": pa.array(labs, type=pa.int64())},
            schema=SCHEMA,
        )
        self._w.write_table(t)
        self.n += len(self._buf)
        self._buf.clear()

    def close(self) -> int:
        self._flush()
        self._w.close()
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def image_bytes(cell) -> bytes | None:
    """从 HF datasets 的 Image 特征里取出原始字节。

    parquet 里它可能是 struct<bytes, path>（dict）或直接被存成 binary。
    """
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return bytes(b)
        p = cell.get("path")
        if p:
            import os
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    return fh.read()
    return None


def read_parquet_rows(path: str | Path):
    """读一个 parquet 分片 -> (列名列表, 逐行 dict 迭代器, 行数)。

    用 read_table 一次性读完（评测集都不大），返回 table 与列名，
    调用方自己按列取值更省内存。
    """
    t = pq.read_table(str(path))
    return t


def count_rows(path: str | Path) -> int:
    return pq.ParquetFile(str(path)).metadata.num_rows


def describe(path: str | Path) -> str:
    """给 --inspect 用：打印 schema + 行数 + 每个 ClassLabel 列的取值分布（前 30 个）。"""
    pf = pq.ParquetFile(str(path))
    lines = [f"{Path(path).name}: {pf.metadata.num_rows} 行, schema = {pf.schema_arrow}"]
    return "\n".join(lines)
