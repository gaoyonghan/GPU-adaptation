"""Apply the recorded NumPy alias compatibility fix for lvis 0.5.3."""

from pathlib import Path
import lvis

path = Path(lvis.__file__).parent / "eval.py"
source = path.read_text()
before, after = "dtype=np.float)", "dtype=float)"
if before in source:
    if source.count(before) != 2:
        raise SystemExit("Unexpected LVIS source; refusing to patch.")
    path.write_text(source.replace(before, after))
elif source.count(after) < 2:
    raise SystemExit("LVIS compatibility patch could not be verified.")
