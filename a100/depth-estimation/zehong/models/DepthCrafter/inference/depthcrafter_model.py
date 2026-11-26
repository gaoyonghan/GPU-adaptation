import torch
import torch.nn as nn
import numpy as np
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
class DepthCrafterWrapper(nn.Module):

    def __init__(self, unet_path: str='tencent/DepthCrafter', pre_train_path: str='stabilityai/stable-video-diffusion-img2vid-xt', num_inference_steps: int=5, guidance_scale: float=1.0, device: torch.device=None):
        super().__init__()
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        self.pipe = DepthCrafterPipeline.from_pretrained(pre_train_path, unet=unet, torch_dtype=torch.float16, variant='fp16')
        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]，由 run_inference 传入
        返回: [B, H, W] 的相对逆深度 (Disparity) Tensor
        """
        b, c, h, w = x.shape
        frames = x.permute(0, 2, 3, 1).cpu().numpy()
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)
        elif frames.ndim == 5:
            frames = frames[0]
        generator = torch.Generator(device='cpu').manual_seed(42)
        res = self.pipe(frames, height=h, width=w, output_type='np', guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps, window_size=1, overlap=0, track_time=False, generator=generator).frames[0]
        if res.ndim == 4:
            res = res[0]
        disp = res.mean(axis=-1)
        disp_tensor = torch.from_numpy(disp).to(self.device, dtype=torch.float32)
        if disp_tensor.ndim == 2:
            disp_tensor = disp_tensor.unsqueeze(0)
        return disp_tensor
