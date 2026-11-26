import argparse, hashlib, io, json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', '--output-dir', type=Path, default=Path('result/run_001'))
    parser.add_argument('--weights', type=Path, default=HERE/'weights')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    # Asset paths are model-relative; absolute paths remain supported.
    for _name in ('weights', 'dataset_root', 'dataset', 'bundle', 'vision_weights', 'hf_cache', 'image'):
        _path = getattr(args, _name, None)
        if isinstance(_path, Path) and not _path.is_absolute():
            setattr(args, _name, (HERE / _path).resolve())
    cfg = json.loads((HERE/'inference/config.json').read_text())
    out = (HERE/args.output_root).resolve()
    if out.exists() or (args.max_samples is not None and args.max_samples < 1):
        parser.error('Use a fresh result directory and a positive sample limit')
    data = args.dataset or (HERE / cfg['dataset']).resolve()
    if not data.is_file():
        parser.error(f'Missing dataset: {data}')
    import torch
    import pyarrow.parquet as pq
    from PIL import Image
    device = args.device or cfg['device']
    if device.startswith('npu'):
        import torch_npu
    if device.split(':')[0] != cfg['device'].split(':')[0]:
        parser.error('Device must use the configured accelerator backend')
    if cfg['kind'] == 'sumsub':
        from inference.diffusions200M import Model200M
        from torchvision import transforms
        model = Model200M()
        model.load_state_dict(torch.load(args.weights/'diffusions200M.pt', map_location='cpu', weights_only=True))
        transform = transforms.Compose([transforms.Resize((640,640)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        def predict(image):
            return model(transform(image).unsqueeze(0).to(device))
        # The publisher pipeline associates raw logit 0 with human and 1 with AI.
        labels = {0:'by human', 1:'by AI'}
    else:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(args.weights, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(args.weights, local_files_only=True)
        labels = model.config.id2label
        def predict(image):
            return model(**processor(images=image, return_tensors='pt').to(device)).logits
    model.to(device).eval()
    if set(labels) != set(cfg['mapping'].values()):
        raise ValueError('Dataset labels do not cover the model label space')
    assert [labels[i].lower() for i in sorted(labels)] == cfg['labels'], labels
    parquet = pq.ParquetFile(data)
    assert parquet.metadata.num_rows == cfg['rows'], 'Unexpected split size'
    limit = args.max_samples or cfg['rows']
    gt, predictions = [], []
    with torch.inference_mode():
        for batch in parquet.iter_batches(batch_size=32, columns=['image',cfg['column']]):
            for row in batch.to_pylist():
                if len(gt) == limit:
                    break
                with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                    logits = predict(image.convert('RGB'))[0].float().cpu()
                if logits.numel() != len(labels) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid model logits')
                ranking = logits.argsort(descending=True, stable=True).tolist()[:5]
                sid = str(len(gt))
                gt.append({'sample_id':sid,'label':cfg['mapping'][str(row[cfg['column']])]})
                predictions.append({'sample_id':sid,'prediction':ranking[0],'topk':ranking})
            if len(gt) == limit:
                break
    assert len(gt) == min(limit,cfg['rows'])
    bundle = out/'scoring_bundle'
    bundle.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    write(bundle/'annotations.json', {'categories':[{'id':i,'name':v} for i,v in labels.items()], 'samples':gt})
    write(bundle/'predictions.json', predictions)
    policy = {'task':'single_label_classification','top_k':[1,5],'top5_when_classes_below_five':'not_applicable',
              'top5_when_ranking_missing':'unavailable','ranking':'preserve_native_order_no_new_tie_break',
              'macro_labels':'observed_ground_truth_classes','zero_division':0,'unit':'fraction','sample_weight':'uniform'}
    write(bundle/'protocol.json', dict(schema='model-eval/score-classification/v1',scoring_ready=True,evaluation=policy,
          model={'id':cfg['repo'],'device':device},dataset={'id':cfg['data_repo'],'split':cfg['split'],
          'scope':'debug_subset' if args.max_samples else 'full_published_split'},
          completed_sample_ids=[r['sample_id'] for r in gt],prediction_count=len(gt),
          artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}))
    write(out/'inference_output.json',dict(state='completed',task='classification',bundles=['scoring_bundle']))

if __name__ == '__main__':
    main()
