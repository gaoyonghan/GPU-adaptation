"""One single-label sample contract over unchanged file/Parquet/TAR/HDF5 assets.

No torch import, preprocessing, extraction, downloads, or model inference here.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import tarfile
from contextlib import ExitStack
from pathlib import Path

SCHEMA = "hengxiang/single-label-dataset/v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Invalid asset-relative path: {name!r}")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError(f"Missing asset or path escape: {name}")
    return result


class StandardDataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "dataset.json").read_text())
        if self.metadata.get("schema") != SCHEMA or self.metadata.get("task") != "single_label_classification":
            raise ValueError("Not a supported single-label dataset manifest")
        for name in ("classes.json", "samples.jsonl"):
            if digest(self.directory / name) != self.metadata["artifact_sha256"][name]:
                raise ValueError(f"Dataset index hash mismatch: {name}")
        self.categories = json.loads((self.directory / "classes.json").read_text())
        ids = [row["id"] for row in self.categories]
        if len(ids) < 2 or any(type(value) is not int for value in ids) or ids != list(range(len(ids))) or any(not isinstance(row.get("name"), str) or not row["name"] for row in self.categories):
            raise ValueError("Category order must explicitly map labels 0..C-1")
        self.root = (self.directory / self.metadata["asset_root"]).resolve()
        if not self.root.is_dir():
            raise ValueError("Dataset asset root missing (materialize local symlink before upload)")

    def records(self):
        seen = set()
        checked_paths = set()
        with (self.directory / "samples.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                sid, label = row.get("sample_id"), row.get("label")
                if not isinstance(sid, str) or not sid or sid in seen:
                    raise ValueError("Invalid/duplicate dataset sample_id")
                if type(label) is not int or not 0 <= label < len(self.categories):
                    raise ValueError("Dataset label outside declared class space")
                if row.get("split") != self.metadata["split"]:
                    raise ValueError("Mixed dataset splits")
                seen.add(sid)
                image = row["image"]
                if image.get("backend") not in {"file", "parquet", "tar", "hdf5"}:
                    raise ValueError("Unsupported image storage backend")
                if image["path"] not in checked_paths:
                    relative_file(self.root, image["path"])
                    checked_paths.add(image["path"])
                if image["backend"] in {"parquet", "hdf5"} and (type(image.get("row")) is not int or image["row"] < 0):
                    raise ValueError("Invalid source row")
                yield row
        if len(seen) != self.metadata["sample_count"]:
            raise ValueError("Dataset index sample count mismatch")

    def validate(self):
        count = sum(1 for _ in self.records())
        return {"schema": SCHEMA, "dataset_id": self.metadata["dataset_id"], "samples": count,
                "classes": len(self.categories), "index_hashes_checked": True,
                "source_asset_content_hashes_recomputed": False, "gpu_used": False}

    def batches(self, batch_size):
        """Yield metadata and RGB images in index order, decoding one batch at a time."""
        from PIL import Image
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        pending_rows, pending_images = [], []
        groups = itertools.groupby(self.records(), key=lambda row: (row["image"]["backend"], row["image"]["path"]))
        with ExitStack() as stack:
            for (backend, name), grouped in groups:
                path = relative_file(self.root, name)
                if backend == "parquet":
                    import pyarrow.parquet as pq
                    rows = iter(grouped)
                    current = next(rows, None)
                    if current is None:
                        continue
                    column = current["image"].get("column", "image")
                    label_column = current["image"].get("label_column")
                    columns = [column] + ([label_column] if label_column else [])
                    index = 0
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
                        values = batch.to_pydict()
                        for offset, value in enumerate(values[column]):
                            if current is not None and current["image"]["row"] < index:
                                raise ValueError("Parquet manifest rows must be unique and increasing per shard")
                            if current is not None and current["image"]["row"] == index:
                                if label_column and values[label_column][offset] != current["label"]:
                                    raise ValueError("Parquet label differs from normalized index")
                                if isinstance(value, dict):
                                    data = value.get("bytes")
                                    source = io.BytesIO(data) if data is not None else relative_file(self.root, value["path"])
                                elif isinstance(value, bytes):
                                    source = io.BytesIO(value)
                                else:
                                    source = relative_file(self.root, value)
                                with Image.open(source) as image:
                                    pending_images.append(image.convert("RGB"))
                                pending_rows.append(current)
                                current = next(rows, None)
                                if len(pending_rows) == batch_size:
                                    yield pending_rows, pending_images
                                    pending_rows, pending_images = [], []
                            index += 1
                    if current is not None:
                        raise ValueError("Parquet row missing")
                    continue
                with ExitStack() as group_stack:
                    archive = group_stack.enter_context(tarfile.open(path, "r:")) if backend == "tar" else None
                    if backend == "hdf5":
                        import h5py
                        hdf5 = group_stack.enter_context(h5py.File(path, "r"))
                    for row in grouped:
                        ref = row["image"]
                        if backend == "file":
                            with Image.open(path) as image:
                                decoded = image.convert("RGB")
                        elif backend == "tar":
                            member = archive.getmember(ref["member"])
                            if not member.isfile():
                                raise ValueError("TAR member is not a regular image")
                            with archive.extractfile(member) as source, Image.open(source) as image:
                                decoded = image.convert("RGB")
                        else:
                            decoded = Image.fromarray(hdf5[ref["key"]][ref["row"]]).convert("RGB")
                        pending_rows.append(row)
                        pending_images.append(decoded)
                        if len(pending_rows) == batch_size:
                            yield pending_rows, pending_images
                            pending_rows, pending_images = [], []
        if pending_rows:
            yield pending_rows, pending_images
