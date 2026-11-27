#!/usr/bin/env python3
"""Prepare pinned WildDet3D sources via GitHub codeload, without git transport."""

from __future__ import annotations

import argparse
import configparser
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


# Submodule SHAs are from the parent's GitHub contents API at this exact commit.
SOURCES = [
    ("WildDet3D", "allenai/WildDet3D", "1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a"),
    ("WildDet3D/third_party/sam3", "weikaih04/sam3_for_wilddet3d", "159490bb13108749dd5ceff68f42cd1792e60641"),
    ("WildDet3D/third_party/lingbot_depth", "Robbyant/lingbot-depth", "ff29de5bf433d1b1519cd851da00bc652668964a"),
    ("_deps/vis4d_cuda_ops", "SysCV/vis4d_cuda_ops", "d2478d80045b62f85be19235981a63ea9bc8db66"),
    ("_deps/utils3d", "EasternJournalist/utils3d", "ca3e9ac015ab3473a37ea60f2f55063696638416"),
]
MARKER = ".official-source.json"


def file_hash(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(directory: Path, revision: str) -> dict:
    marker = directory / MARKER
    if marker.is_file():
        info = json.loads(marker.read_text())
        if info["commit"] != revision or not info.get("files"):
            raise RuntimeError(f"源码版本记录不符: {directory}")
        for relative, expected in info["files"].items():
            path = directory / relative
            if not (path.is_file() or path.is_symlink()) or file_hash(path) != expected:
                raise RuntimeError(f"源码缺失或被修改，保留现场: {path}")
        return {key: value for key, value in info.items() if key != "files"}
    if (directory / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Git 源码版本不符: {directory}, expected={revision}, actual={actual}")
        for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
            subprocess.run(["git", "-C", str(directory), *args], check=True)
        return {"method": "git", "commit": actual}
    raise RuntimeError(f"源码缺少版本记录: {directory}")


def verify_bundle(source_root: Path) -> list[dict]:
    records = [{"path": relative, "repository": repo, **verify_source(source_root / relative, sha)}
               for relative, repo, sha in SOURCES]
    for relative, _, _ in SOURCES:
        config = configparser.ConfigParser()
        config.read(source_root / relative / '.gitmodules')
        for section in config.sections():
            path = relative + '/' + config[section]['path']
            entry = next((record for record in records if record['path'] == path), None)
            if entry is None or config[section]['url'].removesuffix('.git') != 'https://github.com/' + entry['repository']:
                raise RuntimeError(f"存在未锁定的官方子模块: {path}")
    return records


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--show-error", "--silent", "--http1.1",
        "--connect-timeout", "15", "--max-time", "300", "--retry", "2",
        "--retry-all-errors", "--retry-delay", "3", "--output", str(temporary), url,
    ], check=True)
    temporary.replace(destination)


def prepare_archive(source_root: Path, relative: str, repo: str, revision: str) -> None:
    directory = source_root / relative
    if (directory / MARKER).is_file() or (directory / ".git").exists():
        verify_source(directory, revision)
        print(f"复用官方源码: {relative} @ {revision}", flush=True)
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"目标目录含未识别文件，拒绝覆盖: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    cache = source_root / ".archives"
    cache.mkdir(exist_ok=True)
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    archive = cache / f"{repo.replace('/', '__')}-{revision}.tar.gz"
    print(f"准备官方源码归档: {url}", flush=True)
    download(url, archive)
    with tempfile.TemporaryDirectory(prefix=".source-", dir=directory.parent) as temporary:
        with tarfile.open(archive) as handle:
            expected_root = f"{repo.split('/')[-1]}-{revision}"
            if handle.pax_headers.get("comment") != revision:
                raise RuntimeError(f"归档 commit 校验失败: {archive}")
            members = handle.getmembers()
            if any(m.name.split('/')[0] != expected_root or '..' in m.name.split('/') for m in members):
                raise RuntimeError(f"归档根目录校验失败: {archive}")
            # Reject path traversal, escaping links, devices, and unsafe permissions.
            handle.extractall(temporary, filter="data")
        extracted = Path(temporary) / expected_root
        files = {str(path.relative_to(extracted)): file_hash(path)
                 for path in sorted(extracted.rglob('*')) if path.is_file() or path.is_symlink()}
        info = {"method": "github_codeload", "repository": repo, "commit": revision,
                "url": url, "archive_sha256": file_hash(archive), "files": files}
        (extracted / MARKER).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        if directory.exists():
            directory.rmdir()  # Only the empty gitlink directory from its parent archive.
        extracted.rename(directory)
    print(f"官方源码已校验: {relative} @ {revision}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.source_root.mkdir(parents=True, exist_ok=True)
    with (args.source_root / '.source-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not args.check_only:
            for relative, repo, revision in SOURCES:
                prepare_archive(args.source_root, relative, repo, revision)
        print(json.dumps(verify_bundle(args.source_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
