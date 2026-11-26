"""Model-owned configurable project paths; no symlink creation."""
from pathlib import Path
import os

MODEL_DIR=Path(__file__).resolve().parent.parent
PROJECT_ROOT=MODEL_DIR.parents[3]

def project_path(value):
    path=Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT/path).resolve()

DATASET_ROOT=project_path(os.environ.get('CLASSIFICATION_DATASET_ROOT', 'dataset'))
WEIGHTS_PATH=(project_path(os.environ['CLASSIFICATION_WEIGHTS_ROOT'])/MODEL_DIR.name/'best_model.pth'
              if os.environ.get('CLASSIFICATION_WEIGHTS_ROOT') else MODEL_DIR/'weights/best_model.pth')
RESULT_DIR=(project_path(os.environ['CLASSIFICATION_RESULT_ROOT'])/MODEL_DIR.name
            if os.environ.get('CLASSIFICATION_RESULT_ROOT') else MODEL_DIR/'result')
if RESULT_DIR.is_relative_to(PROJECT_ROOT/'classification') and RESULT_DIR != MODEL_DIR/'result':
    raise ValueError('Custom results must not be written inside another model/contributor directory')
