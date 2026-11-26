import torch
from PIL import Image
GRID_SIZE=2
PATCH_SIZE_LV0=512
class PrepareBatch:
    """每张 RGB 图生成固定 2x2 图块及模拟坐标，保持图片 batch 维在最前。"""

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, images):
        batch = []
        coordinates = []
        region_size = GRID_SIZE * PATCH_SIZE_LV0
        for image in images:
            region = image.convert('RGB').resize((region_size, region_size), Image.Resampling.BICUBIC)
            patches = []
            coords = []
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x, y = (col * PATCH_SIZE_LV0, row * PATCH_SIZE_LV0)
                    patch = region.crop((x, y, x + PATCH_SIZE_LV0, y + PATCH_SIZE_LV0))
                    patches.append(self.preprocess(patch))
                    coords.append((x, y))
            batch.append(torch.stack(patches))
            coordinates.append(torch.tensor(coords, dtype=torch.int64))
        return {'images': torch.stack(batch), 'coords': torch.stack(coordinates)}

class TitanImagePipeline(torch.nn.Module):
    """在一次被计时的 forward 中执行图块编码和 TITAN 空间聚合。"""

    def __init__(self, titan, conch):
        super().__init__()
        self.titan = titan
        self.conch = conch

    def forward(self, images, coords):
        batch_size, num_patches = images.shape[:2]
        features = self.conch(images.flatten(0, 1))
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError('CONCH v1.5 应返回 [B * 图块数, D] 的特征 Tensor')
        if features.shape[0] != batch_size * num_patches:
            raise RuntimeError('CONCH v1.5 输出的图块数量不符合输入')
        features = features.reshape(batch_size, num_patches, -1)
        embeddings = []
        for index in range(batch_size):
            embedding = self.titan.encode_slide_from_patch_features(features[index:index + 1], coords[index:index + 1], PATCH_SIZE_LV0)
            if not isinstance(embedding, torch.Tensor):
                raise RuntimeError('TITAN 应返回特征 Tensor')
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise RuntimeError(f'TITAN 单区域输出应为 [1, D]，实际为 {tuple(embedding.shape)}')
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)
