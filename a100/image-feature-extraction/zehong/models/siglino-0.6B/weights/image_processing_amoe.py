import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Union, Dict
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import logging

# Local import of your existing logic
# (Assuming smart_resize and convert_image_to_patches are in the same folder or copied here)
from .image_processor import smart_resize, convert_image_to_patches, pad_along_first_dim

logger = logging.get_logger(__name__)

class AMOEImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "padding_mask", "spatial_shapes"]

    def __init__(
        self,
        patch_size: int = 16,
        min_pixels: int = 128 * 128,
        max_pixels: int = 256 * 256,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

    def preprocess_single(self, image: Image.Image) -> Dict:
        """Standard preprocessing for a single PIL image."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image = image.convert("RGB")
        width, height = image.size # PIL uses (W, H)

        # 1. Smart Resize
        if self.do_resize:
            resized_height, resized_width = smart_resize(
                height, width,
                factor=self.patch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image = image.resize((resized_width, resized_height), Image.BICUBIC)
        else:
            resized_height, resized_width = height, width

        image_np = np.array(image).astype(np.float32)

        # 2. Rescale
        if self.do_rescale:
            image_np = image_np / 255.0

        # 3. Normalize
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)
            std = np.array(self.image_std, dtype=np.float32)
            image_np = (image_np - mean) / std

        spatial_shape = (resized_height // self.patch_size, resized_width // self.patch_size)
        
        # Convert to tensor and patchify
        img_tensor = torch.from_numpy(image_np)
        patches = convert_image_to_patches(img_tensor, self.patch_size)
        
        return {
            "patches": patches,
            "spatial_shape": spatial_shape
        }

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        max_num_patches: int = 256,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> BatchFeature:
        """Main entry point for transformers image processor."""
        if not isinstance(images, (list, tuple)):
            images = [images]

        results = [self.preprocess_single(img) for img in images]
        
        batched_pixels = []
        batched_masks = []
        batched_shapes = []

        for res in results:
            patches = res["patches"]
            shape = res["spatial_shape"]
            
            # Padding logic
            patches_padded, mask = pad_along_first_dim(
                patches,
                max_num_patches,
                pad_value=0.0
            )
            
            batched_pixels.append(patches_padded)
            batched_masks.append(mask)
            batched_shapes.append(list(shape))

        data = {
            "pixel_values": torch.stack(batched_pixels),
            "padding_mask": torch.stack(batched_masks),
            "spatial_shapes": torch.tensor(batched_shapes)
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

