"""NVIDIA入口：官方模型加载 → 结构/环境记录 → 全部图片提取。"""
import os
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
GPU_ID = '0'
MODEL_ID = 'google/hear-pytorch'
BUNDLE_DIR = (Path(__file__).resolve().parents[5] / 'dataset' / 'coswara_cough_1000')
BATCH_SIZE = 64
NUM_WORKERS = 8
WARMUP_STEPS = 3
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU_ID)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'weights/.runtime_cache'))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream'))
import torch
from huggingface_hub import snapshot_download
import numpy as np
import scipy
from transformers import AutoModel
from hear.python.data_processing import audio_utils
from inference.utils.evaluate_model import evaluate_model
from inference.utils.print_model_info import print_model_and_environment
from inference.utils.protocol import checkpoint_fingerprints, sha256_file

class PrepareBatch:
    """CPU逐条进行官方音频预处理；顶层类支持spawn worker。"""

    def __call__(self, audios):
        samples = []
        for sampling_rate, audio in audios:
            audio = audio_utils.resample_audio_and_convert_to_mono(audio_array=audio, sampling_rate=sampling_rate, new_sampling_rate=16000)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or audio.size == 0:
                raise ValueError('Expected nonempty mono audio')
            if not np.isfinite(audio).all():
                raise ValueError('Non-finite audio')
            if np.ptp(audio) == 0:
                raise ValueError('Constant/silent audio')
            if audio.size > 32000:
                power = audio.astype(np.float64) ** 2
                cumulative = np.concatenate(([0.0], np.cumsum(power)))
                starts = np.arange(0, audio.size - 32000 + 1, 160, dtype=np.int64)
                starts = np.unique(np.append(starts, audio.size - 32000))
                energies = cumulative[starts + 32000] - cumulative[starts]
                start = int(starts[np.argmax(energies)])
                audio = audio[start:start + 32000]
            waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            samples.append(audio_utils.preprocess_audio(waveform))
        return {'pixel_values': torch.cat(samples, dim=0)}

def forward_fn(model, inputs):
    return model(**inputs, return_dict=True, output_hidden_states=True)

def select_output(outputs):
    return outputs.pooler_output

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请检查环境；不自动改用CPU')
    snapshot = snapshot_download(MODEL_ID, allow_patterns=['*.safetensors', '*.bin', '*.json', '*.py'])
    model = AutoModel.from_pretrained(snapshot)
    model.cuda().eval()
    output_dir = RUN_ARGS.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print_model_and_environment(model, output_dir)
    summary = evaluate_model(model, PrepareBatch(), forward_fn, select_output, bundle_dir=BUNDLE_DIR, output_dir=output_dir, model_info={'source': 'huggingface', 'id': MODEL_ID, 'revision': Path(snapshot).name, 'checkpoint_files': checkpoint_fingerprints(snapshot)}, preprocessing_info={'class': 'HeAROfficialAudioPreprocessing', 'modality': 'audio', 'sample_unit': 'one_selected_audio_clip', 'dataset': 'Coswara cough-heavy/cough-shallow', 'decode': 'scipy.io.wavfile.read; original PCM/float values', 'resample': 'official resample_audio_and_convert_to_mono; channel mean then scipy.signal.resample to 16000 Hz', 'crop': 'max_energy_32000_samples; hop=160; include_last_window; earliest_tie', 'short_audio': 'official preprocess_audio right-zero-padding', 'spectrogram': 'official preprocess_audio per sample; Mel-PCEN and resize to [1,192,128]', 'audio_utils_sha256': sha256_file(audio_utils.__file__), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__, 'input_hash_protocol': 'named-tensors-original-dtype-le-v2'}, output_name='pooler_output', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, warmup_steps=WARMUP_STEPS)
    print('Results:', output_dir)
    print('Status:', summary['status'], 'images:', summary['num_images'])
    print('Performance:', summary['performance'])
    if summary['status'] != 'complete':
        raise RuntimeError('存在无效输出，请查看summary和feature_index')
RUN_ARGS = None

def snapshot_download(*args, **kwargs):
    return str(RUN_ARGS.weights.resolve())
if __name__ == '__main__':
    main()
