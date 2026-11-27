"""Inherit official text eval; only change initialization and audit/output plumbing."""
import json
import math
import os
from pathlib import Path
from unittest.mock import patch
from vis4d.engine.callbacks import EvaluatorCallback

HERE = Path(__file__).resolve().parent
WEIGHT = Path('/data2/gaoyonghan/gpu_model_standalone_20261110/release/model/检测/nvidia--nemotron-table-structure-v1/resources/model_index/allenai/WildDet3D/wilddet3d_alldata_all_prompt_v1.0.pt')


class RecordedEvaluator(EvaluatorCallback):
    """Record coverage and native metrics without changing native score computation."""
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.image_ids = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs = self.get_test_callback_inputs(outputs, batch)
        self.image_ids.extend(int(i) for i in inputs['coco_image_id'])
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.save_prefix == 'detection_dist' and (batch_idx + 1) % 10 == 0:
            print(f'M27: processed {len(self.image_ids)} images', flush=True)

    def on_test_epoch_end(self, trainer, pl_module):
        run = Path(os.environ['M27_RUN_DIR'])
        expected = json.loads((run / 'expected_ids.json').read_text())
        assert len(self.image_ids) == len(set(self.image_ids)), 'Duplicate evaluated images'
        assert set(self.image_ids) == set(expected), 'Evaluation image coverage mismatch'
        (run / f'{self.save_prefix}_coverage.json').write_text(json.dumps(self.image_ids))
        if int(os.environ['M27_MAX_IMAGES']):
            # A short smoke test is not scored against the full benchmark.
            self.evaluator.reset()
            return
        super().on_test_epoch_end(trainer, pl_module)

    def evaluate(self, metric):
        result = super().evaluate(metric)
        values = {key: float(value) if math.isfinite(float(value)) else None
                  for key, value in result.items()}
        target = Path(os.environ['M27_RUN_DIR']) / f'{self.save_prefix}_{metric}_metrics.json'
        target.write_text(json.dumps(values, indent=2, allow_nan=False))
        return result


def build_geometry(**kwargs):
    from mdm.model.v2 import MDMModel
    from wilddet3d.depth.lingbot_backend import LingbotDepthBackend
    config = json.loads((HERE / 'm27_lingbot_config.json').read_text())['model_config']

    def config_only(cls, path, **unused):
        assert path == 'pretrained/lingbot-depth/postrain-dc-vitl14/model.pt'
        return cls(**config)

    # Same construction as official skip_pretrained, but read the exact official
    # model_config locally instead of downloading a checkpoint just for its config.
    with patch.object(MDMModel, 'from_pretrained', classmethod(config_only)):
        return LingbotDepthBackend(**kwargs)


def load_complete_model():
    import torch
    from vis4d.config import class_config, instantiate_classes
    from configs.eval.in_the_wild.text import get_config as official_config
    config = official_config().model
    config.init_args.sam3_checkpoint = None
    config.init_args.sam3_model = class_config(
        'sam3.model_builder.build_sam3_image_model', checkpoint_path=None,
        load_from_HF=False, device='cpu', eval_mode=False,
        enable_segmentation=False, compile=False)
    config.init_args.geometry_backend.class_path = 'm27_native_config.build_geometry'
    model = instantiate_classes(config)
    saved = torch.load(WEIGHT, map_location='cpu', mmap=True, weights_only=True)
    state = saved.get('state_dict', saved)
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    # No missing key may remain randomly initialized. Do not use strict=False.
    model.load_state_dict(state, strict=True)
    audit = {'checkpoint': str(WEIGHT), 'tensor_count': len(state),
             'missing_keys': [], 'unexpected_keys': [], 'strict_load': True,
             'auxiliary_weights_downloaded': False}
    if os.environ.get('M27_RUN_DIR'):
        (Path(os.environ['M27_RUN_DIR']) / 'weight_load_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'M27: strict checkpoint load passed ({len(state)} tensors), no auxiliary weights required.', flush=True)
    return model


def get_config():
    from vis4d.config import class_config
    from configs.eval.in_the_wild.text import get_config as official_config
    config = official_config()
    config.model = class_config('m27_native_config.load_complete_model')
    # Single GPU only. Keep the official precision and numerical settings.
    config.pl_trainer.strategy = 'auto'
    limit = int(os.environ.get('M27_MAX_IMAGES', '0'))
    if limit:
        config.pl_trainer.limit_test_batches = limit
    for callback in config.callbacks:
        if callback.class_path.endswith('.EvaluatorCallback'):
            callback.class_path = 'm27_native_config.RecordedEvaluator'
    return config
