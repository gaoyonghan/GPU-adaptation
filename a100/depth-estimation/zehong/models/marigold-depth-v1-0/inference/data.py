from pathlib import Path
import cv2,numpy as np
from torch.utils.data import Dataset
DATASET_PROTOCOLS = {'KITTI': {'split': 'Eigen test', 'min_depth': 1e-05, 'max_depth': 80.0, 'crop': 'KITTI benchmark crop followed by Eigen evaluation crop'}, 'NYUv2': {'split': 'test', 'min_depth': 0.001, 'max_depth': 10.0, 'crop': 'Eigen evaluation crop [45:471, 41:601]'}, 'DIODE': {'split': 'val/indoors and val/outdoor', 'min_depth': 0.6, 'max_depth': 350.0, 'crop': None}, 'Sintel': {'split': 'training/final', 'min_depth': 1e-05, 'max_depth': 72.0, 'crop': None}}

def _discover_samples(name, root):
    if name == 'KITTI':
        samples = []
        for depth_path in sorted(root.glob('*_drive_*_sync/proj_depth/groundtruth/image_02/*.png')):
            drive = depth_path.parents[3].name
            date = drive.split('_drive_', 1)[0]
            image_path = root / date / drive / 'image_02' / 'data' / depth_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f'Missing KITTI RGB image: {image_path}')
            samples.append({'id': f'{drive}/{depth_path.stem}', 'image': image_path, 'depth': depth_path})
        return samples
    if name == 'NYUv2':
        list_path = root / 'filename_list_test.txt'
        if not list_path.is_file():
            raise FileNotFoundError(f'Missing NYUv2 test list: {list_path}')
        samples = []
        for line in list_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            image_rel, depth_rel, _ = line.split()
            image_path = root / image_rel
            depth_path = root / depth_rel
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f'Missing NYUv2 pair: {image_path}, {depth_path}')
            samples.append({'id': str(Path(image_rel).with_suffix('')), 'image': image_path, 'depth': depth_path})
        return samples
    if name == 'DIODE':
        samples = []
        for image_path in sorted((root / 'val').glob('**/*.png')):
            depth_path = image_path.with_name(f'{image_path.stem}_depth.npy')
            mask_path = image_path.with_name(f'{image_path.stem}_depth_mask.npy')
            if not depth_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f'Missing DIODE depth or mask: {depth_path}, {mask_path}')
            samples.append({'id': str(image_path.relative_to(root).with_suffix('')), 'image': image_path, 'depth': depth_path, 'mask': mask_path})
        return samples
    if name == 'Sintel':
        samples = []
        image_root = root / 'training' / 'final'
        depth_root = root / 'training' / 'depth'
        for image_path in sorted(image_root.glob('**/*.png')):
            relative_path = image_path.relative_to(image_root)
            depth_path = (depth_root / relative_path).with_suffix('.dpt')
            if not depth_path.is_file():
                raise FileNotFoundError(f'Missing Sintel depth: {depth_path}')
            samples.append({'id': str(relative_path.with_suffix('')), 'image': image_path, 'depth': depth_path})
        return samples
    raise ValueError(f'Unsupported dataset: {name}')

def _read_sintel_depth(path):
    with path.open('rb') as depth_file:
        tag = np.fromfile(depth_file, dtype='<f4', count=1)
        width = np.fromfile(depth_file, dtype='<i4', count=1)
        height = np.fromfile(depth_file, dtype='<i4', count=1)
        if len(tag) != 1 or tag[0] != 202021.25 or len(width) != 1 or (len(height) != 1):
            raise ValueError(f'Invalid Sintel depth header: {path}')
        pixel_count = int(width[0]) * int(height[0])
        depth = np.fromfile(depth_file, dtype='<f4', count=pixel_count)
    if depth.size != pixel_count:
        raise ValueError(f'Truncated Sintel depth file: {path}')
    return depth.reshape((int(height[0]), int(width[0])))

def _kitti_benchmark_crop(array):
    height, width = array.shape[:2]
    crop_height, crop_width = (352, 1216)
    if height < crop_height or width < crop_width:
        raise ValueError(f'KITTI image is smaller than benchmark crop: {array.shape}')
    top = height - crop_height
    left = (width - crop_width) // 2
    return array[top:top + crop_height, left:left + crop_width]

class _DepthEvaluationDataset(Dataset):

    def __init__(self, name, root, max_samples=None):
        self.name = name
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Dataset directory does not exist: {self.root}')
        self.samples = _discover_samples(name, self.root)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f'No evaluation samples found for {name} under {self.root}')

    def __len__(self):
        return len(self.samples)

    def _read_depth_and_mask(self, sample):
        protocol = DATASET_PROTOCOLS[self.name]
        if self.name == 'KITTI':
            depth = cv2.imread(str(sample['depth']), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise OSError(f"Unable to read KITTI depth: {sample['depth']}")
            depth = _kitti_benchmark_crop(depth).astype(np.float32) / 256.0
            valid_mask = np.zeros(depth.shape, dtype=bool)
            height, width = depth.shape
            valid_mask[int(0.3324324 * height):int(0.91351351 * height), int(0.0359477 * width):int(0.96405229 * width)] = True
        elif self.name == 'NYUv2':
            depth = cv2.imread(str(sample['depth']), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise OSError(f"Unable to read NYUv2 depth: {sample['depth']}")
            depth = depth.astype(np.float32) / 1000.0
            valid_mask = np.zeros(depth.shape, dtype=bool)
            valid_mask[45:471, 41:601] = True
        elif self.name == 'DIODE':
            depth = np.load(sample['depth'], allow_pickle=False).squeeze().astype(np.float32)
            valid_mask = np.load(sample['mask'], allow_pickle=False).squeeze() > 0.5
        elif self.name == 'Sintel':
            depth = _read_sintel_depth(sample['depth']).astype(np.float32)
            valid_mask = depth < 10000000000.0
        else:
            raise ValueError(f'Unsupported dataset: {self.name}')
        valid_mask &= np.isfinite(depth)
        valid_mask &= depth > protocol['min_depth']
        valid_mask &= depth < protocol['max_depth']
        return (depth, valid_mask)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample['image']), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unable to read image: {sample['image']}")
        if self.name == 'KITTI':
            image = _kitti_benchmark_crop(image)
        depth, valid_mask = self._read_depth_and_mask(sample)
        if image.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth shape mismatch for {sample['id']}: {image.shape[:2]} vs {depth.shape}")
        return {'index': index, 'id': sample['id'], 'image_path': str(sample['image']), 'depth_path': str(sample['depth']), 'image': cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'output_shape': depth.shape, 'depth': depth, 'valid_mask': valid_mask}

def _collate_as_list(samples):
    return samples
