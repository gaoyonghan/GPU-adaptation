"""Download the fixed author test shard with bounded, validated HTTP ranges."""
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('__DATASET_ROOT__/supplemental_20260905/model30')
URL = 'https://huggingface.co/datasets/yainage90/fashion-object-detection/resolve/959476ce2fde6bd870c6578be22183114b0da131/data/test-00000-of-00001.parquet'
SIZE = 260915347
CHUNK = 4 * 1024 * 1024
ROOT.mkdir(parents=True, exist_ok=True)
started = time.time()
output = ROOT / 'test-00000-of-00001.parquet'
saved = ROOT / 'download.json'
if output.exists() and saved.exists():
    previous = json.loads(saved.read_text())
    if output.stat().st_size == SIZE and hashlib.sha256(output.read_bytes()).hexdigest() == previous['sha256']:
        print('ALREADY_VERIFIED', previous['sha256'], flush=True)
        sys.exit(0)


def fetch(start):
    end = min(start + CHUNK, SIZE) - 1
    target = ROOT / f'part-{start:09d}'
    if target.exists() and target.stat().st_size == end - start + 1:
        return target
    for attempt in range(8):
        try:
            request = urllib.request.Request(URL + f'?range_start={start}', headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=40) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}'
                data = response.read()
            assert len(data) == end - start + 1
            target.write_bytes(data)
            print(f'downloaded {start}-{end}', flush=True)
            return target
        except Exception as error:
            print(f'retry {start} #{attempt + 1}: {error}', flush=True)
            if attempt == 7:
                raise
            time.sleep(1)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    parts = list(pool.map(fetch, range(0, SIZE, CHUNK)))
with output.open('wb') as stream:
    for part in parts:
        stream.write(part.read_bytes())
assert output.stat().st_size == SIZE
sha = hashlib.sha256(output.read_bytes()).hexdigest()
(ROOT / 'download.json').write_text(json.dumps({'url': URL, 'bytes': SIZE, 'sha256': sha, 'elapsed_seconds': time.time() - started}, indent=2))
print('COMPLETE', sha, flush=True)
