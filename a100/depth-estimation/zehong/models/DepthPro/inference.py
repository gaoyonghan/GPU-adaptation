from pathlib import Path
import argparse, csv, json, time, sys
from dataclasses import replace

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['NYUv2'], default='NYUv2')
    p.add_argument('--dataset-root', type=Path, default=HERE.parents[3]/'dataset')
    p.add_argument('--weights', type=Path, default=HERE/'weights')
    p.add_argument('--output-root', '--output-dir', type=Path, default=Path('result/run_001'))
    p.add_argument('--max-samples', type=int)
    p.add_argument('--warmup-runs', '--warmup', type=int, default=3)
    a = p.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(a, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(a, _name, (HERE / _path).resolve())
    if a.warmup_runs < 0:
        p.error('--warmup-runs must be nonnegative')
    out = (HERE/a.output_root).resolve()
    if out.exists() or (a.max_samples is not None and a.max_samples < 1):
        p.error('Use a fresh output directory and a positive sample limit')
    root = (HERE/a.dataset_root).resolve()
    weights = (HERE/a.weights).resolve()
    pairs = [line.split()[:2] for line in (root/'NYUv2/filename_list_test.txt').read_text().splitlines() if line.strip()]
    if len(pairs) != 654 or len({pair[0] for pair in pairs}) != 654:
        raise ValueError('Expected 654 distinct NYUv2 test images')
    for pair in pairs:
        for path in pair:
            if not (root/'NYUv2'/path).is_file(): raise FileNotFoundError(path)
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    if False:  # Backend belongs to this platform, not its mount path.
        import torch_npu
        device = torch.device('npu:0')
    else:
        device = torch.device('cuda:0')
    api = getattr(torch,device.type)
    if not api.is_available(): raise RuntimeError(f'{device} unavailable')
    api.set_device(device)
    if HERE.name == 'DepthPro-hf':
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(weights,local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(weights,local_files_only=True).to(device).eval()
        def prepare(image): return processor(images=image,return_tensors='pt').to(device)
        def forward(inputs): return model(**inputs)
        def recover(value,size): return processor.post_process_depth_estimation(value,target_sizes=[size])[0]['predicted_depth']
    else:
        sys.path.insert(0,str(HERE/'inference/upstream'))
        from depth_pro.depth_pro import create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT
        model, transform = create_model_and_transforms(replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(weights/'depth_pro.pt')),device=device)
        model.eval()
        def prepare(image): return transform(np.array(image)).to(device)
        def forward(inputs): return model.infer(inputs,f_px=None)['depth']
        def recover(value,size): return value
    # The native infer method directly calls model.forward between its
    # resize and metric-depth conversion. Measure that boundary in both versions.
    network_forward = model.forward
    begin = api.Event(enable_timing=True)
    end = api.Event(enable_timing=True)

    def timed_forward(*args, **kwargs):
        begin.record()
        result = network_forward(*args, **kwargs)
        end.record()
        return result

    timing_kind = f'{device.type}_events_model_forward'
    dest = out/'NYUv2'
    (dest/'predictions').mkdir(parents=True)
    def write(name,value): (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    write('inference_output.json',{'state':'running','task':'depth-estimation'})
    try:
        records=[]
        with torch.inference_mode():
            if a.warmup_runs:
                with Image.open(root/'NYUv2'/pairs[0][0]) as first_image:
                    warm_inputs = prepare(first_image.convert('RGB'))
                for _ in range(a.warmup_runs):
                    forward(warm_inputs)
                api.synchronize()
                del warm_inputs

            model.forward = timed_forward
            start = time.perf_counter()
            for i,(image,gt) in enumerate(pairs[:a.max_samples]):
                img=Image.open(root/'NYUv2'/image).convert('RGB')
                size=cv2.imread(str(root/'NYUv2'/gt),cv2.IMREAD_UNCHANGED).shape
                inputs=prepare(img)
                value = forward(inputs)
                end.synchronize()
                ms = begin.elapsed_time(end)
                depth=recover(value,size).float().cpu().numpy().squeeze()
                if depth.shape!=size: depth=cv2.resize(depth,(size[1],size[0]),interpolation=cv2.INTER_LINEAR)
                if not np.isfinite(depth).all() or (depth<=0).any(): raise ValueError(f'Invalid metric depth for {image}')
                raw=f'NYUv2/predictions/{i:04d}.npy'
                np.save(out/raw,depth)
                records.append(dict(sample_id=image,image_path=str(root/'NYUv2'/image),depth_path='NYUv2/'+gt,raw_prediction_path=raw,height=size[0],width=size[1],forward_ms=ms,forward_timing_kind=timing_kind))
        with (dest/'inference_records.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        elapsed=time.perf_counter()-start
        times=np.array([r['forward_ms'] for r in records])
        timing=dict(sample_count=len(records),batch_size=1,warmup_runs=a.warmup_runs,timing_backend=timing_kind,total_forward_seconds=float(times.sum()/1000),stage1_dataset_seconds=elapsed,stage1_dataset_fps=len(records)/elapsed,mean_forward_ms=float(times.mean()),median_forward_ms=float(np.median(times)),p95_forward_ms=float(np.percentile(times,95)),forward_fps=1000/float(times.mean()))
        write('evaluation_config.json',dict(datasets=['NYUv2'],dataset_root=str(root),prediction_type='metric depth',max_samples=a.max_samples,device=str(device),batch_size=1,warmup_runs=a.warmup_runs,timing={'backend':timing_kind,'forward':'Device events around model.forward after input preparation and before metric depth recovery or resizing','end_to_end':'Timed dataset iteration through prediction and CSV saving, after warmup'},focal_length='model estimated',evaluation_scope='functional_subset' if a.max_samples else 'full_native_split'))
        write('inference_summary.json',dict(datasets={'NYUv2':timing},overall=timing))
        write('inference_output.json',dict(state='completed',task='depth-estimation',datasets=['NYUv2']))
    except Exception:
        write('inference_output.json',dict(state='failed',task='depth-estimation'))
        raise

if __name__=='__main__':
    main()
