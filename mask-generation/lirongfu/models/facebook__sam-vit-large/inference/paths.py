"""Model-owned paths; persisted paths are relative to the project root."""
from pathlib import Path
import os

MODEL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODEL_DIR.parents[3]

def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

DATASET_ROOT = project_path(os.environ.get('MASK_DATASET_ROOT', 'dataset'))
WEIGHTS_PATH = (project_path(os.environ['MASK_WEIGHTS_ROOT']) / MODEL_DIR.name / 'best_model.pth'
                if os.environ.get('MASK_WEIGHTS_ROOT') else MODEL_DIR / 'weights/best_model.pth')
RESULT_DIR = (project_path(os.environ['MASK_RESULT_ROOT']) / MODEL_DIR.name
              if os.environ.get('MASK_RESULT_ROOT') else MODEL_DIR / 'result')
EVALUATOR = PROJECT_ROOT / 'mask-generation/evaluate.py'

def resolve_legacy_dataset(relative):
    prefix = '../../Dataset/'
    if not isinstance(relative, str) or not relative.startswith(prefix):
        raise ValueError('Unexpected original dataset mapping')
    target = (DATASET_ROOT / relative[len(prefix):]).resolve(strict=True)
    if not target.is_relative_to(DATASET_ROOT.resolve()):
        raise ValueError('Dataset mapping escapes the configured root')
    return target
