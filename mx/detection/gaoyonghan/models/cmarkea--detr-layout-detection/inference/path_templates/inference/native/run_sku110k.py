"""SKU110K decoding repair: retain truncated JPEGs and log every tolerant decode."""
import argparse
import json
from pathlib import Path
from PIL import ImageFile
import run_model as runner

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-images',type=int,default=10);p.add_argument('--device',default='cuda:1');args=p.parse_args()
    args.output_root=Path('__MODEL_ROOT__/resources/runtime/assets/workspace/results/remaining25_v5');args.dataset_root=runner.core.DEFAULT_DATASET_ROOT
    manifest=json.loads((runner.HERE/'models.json').read_text());model=next(m for m in manifest['models'] if m['sequence']==54);task=model['tasks'][0]
    task['protocol_overrides']={'jpeg_decoder_policy':'Retry only PIL truncated-image errors with LOAD_TRUNCATED_IMAGES; keep every annotated test image; record each affected image.', 'decoder_wrapper_sha256':runner.file_hash(Path(__file__))}
    prepare=runner.adapters.prepare
    def with_decoder(*a):
        prepared=prepare(*a);read=prepared.read_image
        def read_image(info):
            try:return read(info)
            except OSError as exc:
                if 'truncated' not in str(exc):raise
                try:
                    ImageFile.LOAD_TRUNCATED_IMAGES=True
                    image=read(info)
                finally:ImageFile.LOAD_TRUNCATED_IMAGES=False
                runner.core.append_jsonl(args.output_root/model['model_key']/task['key']/'decoder_repairs.jsonl',{'image_id':info['id'],'file_name':info['file_name'],'error':str(exc),'action':'PIL tolerant decode; no labels or dimensions changed'})
                return image
        prepared.read_image=read_image
        return prepared
    runner.adapters.prepare=with_decoder
    return 0 if runner.run_task(model,task,manifest,args) else 1

if __name__=='__main__':raise SystemExit(main())
