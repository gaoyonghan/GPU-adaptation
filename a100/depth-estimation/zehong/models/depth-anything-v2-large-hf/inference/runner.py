"""NYUv2适配；前向仅model(pixel_values=...)，后处理与评分独立。"""
import os
import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def _readable_timing_fields(value):
    """Add readable aliases; preserve measured timing boundaries and legacy keys."""
    if not isinstance(value, dict):
        return value
    for child in value.values():
        if isinstance(child, dict):
            _readable_timing_fields(child)
    if 'mean_forward_ms' in value:
        value['forward_ms_per_image'] = value['mean_forward_ms']
    if 'forward_fps' in value:
        value['forward_images_per_second'] = value['forward_fps']
    if 'total_forward_seconds' in value:
        value['forward_seconds'] = value['total_forward_seconds']
    datasets = value.get('datasets')
    if isinstance(datasets, dict) and datasets:
        rows = list(datasets.values())
        rows = [r.get('evaluation', r) for r in rows]
        counts = [r.get('forward_recorded_images', r.get('stage1_completed_images', r.get('num_samples', r.get('sample_count')))) for r in rows]
        seconds = [r.get('total_forward_seconds', r.get('forward_seconds')) for r in rows]
        if all(n is not None for n in counts) and all(s is not None for s in seconds):
            n, total = sum(counts), sum(seconds)
            overall = value.setdefault('overall', {})
            info = overall.get('evaluation', overall)
            info['forward_seconds'] = total
            info['forward_ms_per_image'] = total * 1000 / n if n else None
            info['forward_images_per_second'] = n / total if total > 0 else None
    return value

def _discover_samples(name, root):
    if name == "KITTI":
        samples = []
        for depth_path in sorted(root.glob("*_drive_*_sync/proj_depth/groundtruth/image_02/*.png")):
            drive = depth_path.parents[3].name
            date = drive.split("_drive_", 1)[0]
            image_path = root / date / drive / "image_02" / "data" / depth_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing KITTI RGB image: {image_path}")
            samples.append(
                {
                    "id": f"{drive}/{depth_path.stem}",
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "NYUv2":
        list_path = root / "filename_list_test.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing NYUv2 test list: {list_path}")
        samples = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_rel, depth_rel, _ = line.split()
            image_path = root / image_rel
            depth_path = root / depth_rel
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Missing NYUv2 pair: {image_path}, {depth_path}")
            samples.append(
                {
                    "id": str(Path(image_rel).with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "DIODE":
        samples = []
        for image_path in sorted((root / "val").glob("**/*.png")):
            depth_path = image_path.with_name(f"{image_path.stem}_depth.npy")
            mask_path = image_path.with_name(f"{image_path.stem}_depth_mask.npy")
            if not depth_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing DIODE depth or mask: {depth_path}, {mask_path}"
                )
            samples.append(
                {
                    "id": str(image_path.relative_to(root).with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                    "mask": mask_path,
                }
            )
        return samples

    if name == "Sintel":
        samples = []
        image_root = root / "training" / "final"
        depth_root = root / "training" / "depth"
        for image_path in sorted(image_root.glob("**/*.png")):
            relative_path = image_path.relative_to(image_root)
            depth_path = (depth_root / relative_path).with_suffix(".dpt")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing Sintel depth: {depth_path}")
            samples.append(
                {
                    "id": str(relative_path.with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    raise ValueError(f"Unsupported dataset: {name}")



class NYUImages(Dataset):
    def __init__(self, root, processor, name, max_samples):
        self.root, self.processor, self.name = Path(root), processor, name
        self.samples = _discover_samples(name, self.root/name)
        if max_samples is not None: self.samples = self.samples[:max_samples]
        if not self.samples: raise ValueError('No evaluation images found')
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        sample = self.samples[i]
        import cv2
        if self.processor is None:
            from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
            image = cv2.imread(str(sample['image']))
            if image is None: raise OSError(sample['image'])
            if self.name == 'KITTI': image = image[-352:, (image.shape[1]-1216)//2:(image.shape[1]-1216)//2+1216]
            shape = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
            transform = [Resize(width=518,height=518,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()]
            data={'image':rgb}
            for fn in transform: data=fn(data)
            inputs=torch.from_numpy(data['image'])
        else:
            with Image.open(sample['image']) as opened:
                image=opened.convert('RGB')
                if self.name == 'KITTI':
                    left=(image.width-1216)//2
                    image=image.crop((left,image.height-352,left+1216,image.height))
                shape=(image.height,image.width)
                inputs=self.processor(images=image,return_tensors='pt')['pixel_values'][0]
        return inputs, {'sample_id':sample['id'],'image_path':str(sample['image'].relative_to(self.root)),
            'depth_path':str(sample['depth'].relative_to(self.root)),
            'mask_path':str(sample['mask'].relative_to(self.root)) if 'mask' in sample else '', 'shape':shape}


def collate(samples): return samples


def run(model, processor, root, output, workers, warmup_steps, name, max_samples):
    output, root = Path(output), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    cuda = device.type == 'cuda'
    dataset = NYUImages(root, processor, name, max_samples)
    # 预热不进入前向和提取端到端计时。
    first = dataset[0][0].unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(warmup_steps): model(pixel_values=first) if processor is not None else model(first)
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    del first
    loader = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False,
                        collate_fn=collate, pin_memory=cuda,
                        multiprocessing_context="spawn" if workers > 0 else None)
    records, timings = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for samples in loader:
            cpu_input, sample = samples[0]
            inputs = cpu_input.unsqueeze(0).to(device)
            if cuda:
                # 同一stream：输入搬运先完成；只包围官方模型调用。
                begin.record()
                outputs = model(pixel_values=inputs) if processor is not None else model(inputs)
                end.record(); end.synchronize()
                elapsed = begin.elapsed_time(end)
            else:
                tick = time.perf_counter()
                outputs = model(pixel_values=inputs) if processor is not None else model(inputs)
                elapsed = (time.perf_counter()-tick)*1000
            # 输出字段选择、插值、D2H、检查、保存均在计时外。
            prediction = outputs.predicted_depth if processor is not None else outputs
            prediction = F.interpolate(prediction[:,None], size=sample['shape'], mode='bicubic' if processor is not None else 'bilinear', align_corners=processor is None)[0,0]
            array = prediction.float().cpu().numpy()
            if not np.isfinite(array).all(): raise ValueError('预测包含NaN/Inf')
            relative = Path(name)/'raw_predictions'/(sample['sample_id']+'.npy')
            (output/relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(output/relative,array)
            records.append(dict(dataset=name, sample_id=sample['sample_id'],
                image_path=sample['image_path'], depth_path=sample['depth_path'],
                mask_path=sample['mask_path'], raw_prediction_path=str(relative), height=sample['shape'][0], width=sample['shape'][1],
                input_height=int(inputs.shape[-2]), input_width=int(inputs.shape[-1]),
                forward_ms=elapsed, forward_timing_kind='single_image'))
            timings.append(elapsed)
            print(f'Inference {len(records)}/{len(dataset)}',flush=True)
    with (output/name/'inference_records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    if cuda: torch.cuda.synchronize(device)
    e2e = time.perf_counter()-start
    total = sum(timings)/1000
    info=dict(device=str(device),num_samples=len(records),stage1_completed_images=len(records),
        forward_recorded_images=len(records),stage1_status='completed',accuracy_status='not_run',
        total_forward_seconds=total,mean_forward_ms=float(np.mean(timings)),median_forward_ms=float(np.median(timings)),
        p95_forward_ms=float(np.percentile(timings,95)),forward_fps=len(records)/total,
        stage1_dataset_seconds=e2e,stage1_dataset_fps=len(records)/e2e,
        timing_backend='cuda_event' if cuda else 'cpu_wall')
    config=dict(datasets=[name],prediction_type='affine-invariant inverse depth',max_samples=max_samples,
        dataset_root=os.path.relpath(root, output.resolve()), evaluation_scope='functional_subset' if max_samples is not None else 'full_dataset',
        preprocessing=processor.to_dict() if processor is not None else {'source':'official native OpenCV transform','input_size':518},warmup_runs=warmup_steps,batch_size=1,num_workers=workers,
        parameter_dtype=str(next(model.parameters()).dtype),
        timing={'forward':'model(pixel_values=inputs) only; output selection/interpolation/D2H/save excluded',
                'end_to_end':'DataLoader iteration through prediction/CSV save; excludes loading/warmup/scoring'})
    (output/'evaluation_config.json').write_text(json.dumps(_readable_timing_fields(config),indent=2))
    (output/'inference_summary.json').write_text(json.dumps(_readable_timing_fields({'datasets':{name:info},'overall':{'stage1_completed_images':len(records)}}),indent=2))
    (output/'model_structure.txt').write_text(str(model))
    (output/'model_environment.json').write_text(json.dumps(_readable_timing_fields({'torch':torch.__version__,'device':str(device),'python':__import__('sys').version}),indent=2))
    print(json.dumps(_readable_timing_fields(info),indent=2))
