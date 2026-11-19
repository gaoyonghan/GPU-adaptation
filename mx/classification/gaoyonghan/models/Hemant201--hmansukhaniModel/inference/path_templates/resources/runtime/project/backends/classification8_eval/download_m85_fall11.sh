#!/usr/bin/env bash
set -euo pipefail

# Third-party Fall 2011 archive; no inference, extraction, or model changes.
# P2P exposes the peer IP and may upload pieces while downloading.
DEST=__DATASET_ROOT__/imagenet22k_fall11
HASH=564a77c1e1119da199ff32622a1609431b9f1c47
for tool in aria2c curl flock; do
    command -v "$tool" >/dev/null || { echo "缺少命令: $tool" >&2; exit 1; }
done
mkdir -p "$DEST"
exec 9>"$DEST/.download.lock"
flock -n 9 || { echo '该目录已有下载进程。' >&2; exit 1; }
cd "$DEST"
if [[ ! -s fall11_whole.torrent ]]; then
    curl --fail --location --retry 5 --connect-timeout 30 --max-time 300 \
        "https://academictorrents.com/download/$HASH.torrent" \
        -o fall11_whole.torrent.part
    aria2c --no-conf=true --show-files=true fall11_whole.torrent.part
    mv fall11_whole.torrent.part fall11_whole.torrent
fi
INFO=$(LC_ALL=C aria2c --no-conf=true --show-files=true fall11_whole.torrent)
printf '%s\n' "$INFO"
if [[ "${INFO,,}" != *"$HASH"* ]]; then
    echo '种子 info hash 与预期不符，停止。' >&2
    exit 1
fi
echo '下载约1.31 TB；支持中断续传。下载期间可能上传分块（上限1 MiB/s），完成后不继续做种。'
aria2c --no-conf=true --continue=true --check-integrity=true \
    --file-allocation=none --auto-file-renaming=false \
    --seed-time=0 --max-overall-upload-limit=1M \
    --summary-interval=60 --dir="$DEST" fall11_whole.torrent
[[ $(stat -c %s "$DEST/fall11_whole.tar") == 1309848811520 ]] || {
    echo '归档大小与存档页面不符，请勿评测。' >&2; exit 1;
}
echo '下载完成并通过种子分块校验；尚需核查官方验证清单覆盖与图像完整性。先不要全量解压。'
