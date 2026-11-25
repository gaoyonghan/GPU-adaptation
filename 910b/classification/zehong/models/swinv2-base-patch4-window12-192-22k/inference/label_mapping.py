import json,os,re
from pathlib import Path
def _synset_from_value(value):
    """Return a WordNet synset (for example, n01440764) from a label value."""
    match = re.search('\\\\bn\\\\d{8}\\\\b', str(value))
    return match.group(0) if match else None

def _build_imagenet1k_to_22k_mapping(model, synset_mapping_path):
    """Load the local official Swin 22K-to-1K logit-column mapping.

    Microsoft Swin's ``data/map22kto1k.txt`` contains exactly 1,000 integers.
    Line ``i`` is the original 21,841-way classifier-output column for
    ImageNet-1K label ``i``.  Thus it is sufficient for both open-set 22K
    matching and 1K-restricted matching; no ImageNet-22K images are required.

    A JSON mapping is also accepted for compatibility with the documented
    synset-based format used previously.
    """
    configured_path = synset_mapping_path or os.environ.get('IMAGENET_1K_TO_22K_MAPPING_PATH')
    if not configured_path:
        packaged_path = Path(__file__).with_name('map22kto1k.txt')
        if packaged_path.is_file():
            configured_path = packaged_path
    if not configured_path:
        raise FileNotFoundError('Missing local 1K-to-22K synset mapping. Set IMAGENET_1K_TO_22K_MAPPING_PATH or pass synset_mapping_path=... to evaluate_model(). Put the official Swin data/map22kto1k.txt beside this file to load it automatically. The local ImageNet-1K images alone do not define the 22K classifier-column order.')
    path = Path(configured_path)
    if not path.is_file():
        raise FileNotFoundError(f'1K-to-22K synset mapping file not found: {path}')
    if int(getattr(model.config, 'num_labels', 0)) != 21841:
        raise ValueError('Expected the original 21,841-class SwinV2-22K classifier head')
    text = path.read_text(encoding='utf-8').strip()
    if path.suffix.lower() == '.json':
        data = json.loads(text)
        imagenet1k = data.get('imagenet1k_index_to_synset')
        model_synsets = data.get('imagenet22k_synset_to_index')
        if not isinstance(imagenet1k, dict) or not isinstance(model_synsets, dict):
            raise ValueError('Invalid synset mapping JSON. Required keys: imagenet1k_index_to_synset and imagenet22k_synset_to_index.')
        try:
            indices = [int(model_synsets[imagenet1k[str(index)]]) for index in range(1000)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('The JSON mapping must define 1,000 valid 1K-to-22K entries') from error
        model_synset_count = len(model_synsets)
        mapping_type = 'ImageNet-1K index -> WordNet synset -> checkpoint 22K logit index'
    else:
        try:
            indices = [int(line.strip()) for line in text.splitlines() if line.strip()]
        except ValueError as error:
            raise ValueError(f'Invalid integer in official map22kto1k file: {path}') from error
        if len(indices) != 1000:
            raise ValueError(f'Official map22kto1k.txt must contain 1,000 indices; found {len(indices)}')
        model_synset_count = None
        mapping_type = 'Official Swin map22kto1k.txt: ImageNet-1K index -> checkpoint 22K logit index'
    if any((index < 0 or index >= model.config.num_labels for index in indices)):
        raise ValueError('A 22K logit index is outside the model output range')
    if len(set(indices)) != 1000:
        raise ValueError('ImageNet-1K synsets do not map to 1,000 distinct 22K logit columns')
    return (indices, {'checked': True, 'mapping_type': mapping_type, 'source': str(path.resolve()), 'mapped_class_count': len(indices), 'model_synset_count': model_synset_count, 'first_mappings': [{'imagenet1k_index': i, 'model_22k_index': indices[i]} for i in range(min(10, len(indices)))]})