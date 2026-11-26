"""Apply the verified empty-mask fix to the private Transformers installation only."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'_ppdeps/transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3_fast.py'
record=json.loads((root/'assets/pp_v3_patch/patch.json').read_text());source=path.read_text()
current=hashlib.sha256(source.encode()).hexdigest()
if current==record['patched_sha256']:
    print('PP V3 private patch already present')
    raise SystemExit(0)
assert current==record['original_sha256'],'Unexpected upstream source; do not patch another version'
fixed=source.replace('            # resize mask to match box size','            if cropped_mask.size == 0:\n                polygon_points.append(rect)\n                continue\n\n            # resize mask to match box size').replace('            polygon_points.append(polygon)\n','            polygon_points.append(polygon if polygon is not None else rect)\n')
assert hashlib.sha256(fixed.encode()).hexdigest()==record['patched_sha256']
path.write_text(fixed)
