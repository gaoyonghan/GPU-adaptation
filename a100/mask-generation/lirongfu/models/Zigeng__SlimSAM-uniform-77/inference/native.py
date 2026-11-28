#!/usr/bin/env python3
"""Offline mask inference -> shared task evaluator -> this model's result directory.

Native original weights/configuration are preserved in best_model.pth; model
sources are embedded below. Dataset paths are relative to this model directory.
Work/checkpoints live outside Model and Dataset, survive interruption, and are
removed after successful evaluation unless --keep-work is selected.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

MODEL_SPEC = {'backend': 'sam_hf', 'dataset': 'SA-1B-val', 'count': 1133700, 'batch_size': 64, 'weights': ['model.safetensors'], 'key': 'Zigeng__SlimSAM-uniform-77'}
EMBEDDED_SOURCES = {}
NODE_SOURCE = "// Native Transformers.js inference only. Python evaluate.py loads original GT.\nimport fs from 'node:fs';\nimport path from 'node:path';\nimport readline from 'node:readline';\nimport {AutoProcessor, RawImage, SamModel, Sam3TrackerModel, Tensor} from '@huggingface/transformers';\n\nconst [modelDir, datasetRoot, manifest, output, backend, device, batchArg] = process.argv.slice(2);\nlet safeBatch = Number(batchArg);\nif (!Number.isInteger(safeBatch) || safeBatch < 1) throw new Error('Invalid batch size');\nconst completed = new Set();\nif (fs.existsSync(output)) {\n  for await (const line of readline.createInterface({input: fs.createReadStream(output), crlfDelay: Infinity})) {\n    if (line.trim()) completed.add(JSON.parse(line).sample_id);\n  }\n}\nconsole.log(`Resume: ${completed.size} completed instances; device=${device}; batch=${safeBatch}`);\nconst processor = await AutoProcessor.from_pretrained(modelDir, {local_files_only:true});\nconst Class = backend === 'sam3_onnx' ? Sam3TrackerModel : SamModel;\nconst model = await Class.from_pretrained(modelDir, {local_files_only:true, device, dtype:'fp32'});\n// Native ORT run resolves after host outputs are ready, including CUDA work.\n// Wrap graph execution only, outside processor calls and mask decoding.\nlet forwardSeconds = 0;\nfor (const session of Object.values(model.sessions)) {\n  const originalRun = session.run.bind(session);\n  session.run = async (...args) => {\n    const begin = process.hrtime.bigint();\n    const result = await originalRun(...args);\n    forwardSeconds += Number(process.hrtime.bigint() - begin) / 1e9;\n    return result;\n  };\n}\nconst handle = fs.openSync(output, 'a');\nlet processed = 0, seen = completed.size;\nconst started = Date.now();\nconst dispose = values => {for (const v of values) v?.dispose?.();};\nfunction encode(values, offset, height, width) {\n  const counts = [];\n  let previous = false, count = 0;\n  // COCO RLE is column-major; postprocessed ONNX tensors are row-major.\n  for (let x=0; x<width; x++) for (let y=0; y<height; y++) {\n    const bit = values[offset+y*width+x] > 0;\n    if (bit===previous) count++;\n    else {counts.push(count);count=1;previous=bit;}\n  }\n  counts.push(count);\n  return {size:[height,width],counts};\n}\nasync function inferGroup(group) {\n  const pending = group.filter(x => !completed.has(x.sample_id));\n  if (!pending.length) return;\n  const image = await RawImage.read(path.join(datasetRoot,group[0].image));\n  const inputs = await processor(image);\n  forwardSeconds = 0;\n  const embeddings = await model.get_image_embeddings(inputs);\n  let encoderSeconds = forwardSeconds;\n  inputs.pixel_values.dispose?.();\n  for (let begin=0; begin<pending.length;) {\n    const batch = pending.slice(begin,begin+safeBatch);\n    const points = processor.reshape_input_points([batch.map(x=>[[...x.point]])],inputs.original_sizes,inputs.reshaped_input_sizes);\n    const labels = processor.image_processor.add_input_labels([batch.map(()=>[1])],points);\n    let out;\n    forwardSeconds = 0;\n    try {out = await model({...embeddings,input_points:points,input_labels:labels});}\n    catch(error) {\n      dispose([points,labels]);\n      if (batch.length>1 && /Failed to allocate memory|out of memory|CUDA.*memory/i.test(String(error?.stack??error))) {\n        safeBatch=Math.max(1,Math.floor(batch.length/2));\n        console.warn(`Retrying same instances with prompt batch ${safeBatch}`);\n        continue;\n      }\n      throw error;\n    }\n    const elapsedForward = encoderSeconds + forwardSeconds;\n    const scores=out.iou_scores.data, candidates=out.iou_scores.dims.at(-1);\n    const h=out.pred_masks.dims.at(-2),w=out.pred_masks.dims.at(-1),area=h*w;\n    const selected = new Float32Array(batch.length*area);\n    const bestCandidates = new Int32Array(batch.length);\n    for (let i=0;i<batch.length;i++) {\n      let best=0;\n      for(let k=1;k<candidates;k++) if(scores[i*candidates+k]>scores[i*candidates+best]) best=k;\n      bestCandidates[i]=best;\n      selected.set(out.pred_masks.data.subarray((i*candidates+best)*area,(i*candidates+best+1)*area),i*area);\n    }\n    // Preserve each previously tested backend's postprocessing tensor shape.\n    const tracker = backend === 'sam3_onnx';\n    const low = tracker ? null : new Tensor('float32',selected,[1,batch.length,1,h,w]);\n    const full = (await processor.post_process_masks(tracker ? out.pred_masks : low,inputs.original_sizes,inputs.reshaped_input_sizes,{binarize:tracker}))[0];\n    const height=full.dims.at(-2),width=full.dims.at(-1);\n    for(let i=0;i<batch.length;i++) {\n      const offset = (tracker ? i*candidates+bestCandidates[i] : i)*height*width;\n      const record={sample_id:batch[i].sample_id,original:batch[i].original,\n                    prediction:encode(full.data,offset,height,width),forward_latency_s:elapsedForward/batch.length};\n      fs.writeSync(handle,JSON.stringify(record)+'\\n');\n      completed.add(record.sample_id);processed++;seen++;\n    }\n    fs.fsyncSync(handle);\n    const rate=processed/((Date.now()-started)/1000);\n    process.stdout.write(`\\r${seen} instances; ${rate.toFixed(2)} instances/s`);\n    dispose([points,labels,out.pred_masks,out.iou_scores,out.object_score_logits,low,full]);\n    encoderSeconds=0;\n    begin+=batch.length;\n  }\n  dispose(Object.values(embeddings));\n}\ntry {\n  let group=[];\n  for await(const line of readline.createInterface({input:fs.createReadStream(manifest),crlfDelay:Infinity})) {\n    if(!line.trim()) continue;\n    const item=JSON.parse(line);\n    if(group.length && group[0].image!==item.image) {await inferGroup(group);group=[];}\n    group.push(item);\n  }\n  if(group.length) await inferGroup(group);\n} finally {fs.fsyncSync(handle);fs.closeSync(handle);}\nconsole.log(`\\nCompleted ${seen} instances; predictions saved for Python mIoU evaluation.`);\n// Avoid a known ORT CUDA-provider destructor failure after synced output.\nprocess.exit(0);\n"
import uuid
_paths_spec = importlib.util.spec_from_file_location('_mask_model_paths', Path(__file__).with_name('paths.py'))
PATHS = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(PATHS)
HERE = PATHS.MODEL_DIR


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative_member(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or any(x in ("", ".", "..") for x in name.split("/")) or "\\" in name:
        raise ValueError(f"Unsafe bundle member: {name}")
    return root.joinpath(*path.parts)


def atomic_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def load_metric():
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('_shared_mask_evaluator', PATHS.EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The native runner wraps this proxy's evaluate method. The shared module's
    # own evaluate function must remain untouched for standalone COCO scoring.
    return SimpleNamespace(**{key: value for key, value in vars(module).items() if not key.startswith('__')})


def maybe_switch_environment(args):
    wanted = MODEL_SPEC.get("environment", "mask-eval")
    if args.no_env_switch or args.check or sys.prefix.split(os.sep)[-1] == wanted:
        return
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    candidates += [str(p / "bin/conda") for p in Path(sys.executable).resolve().parents]
    conda = next((p for p in candidates if p and Path(p).is_file()), None)
    if conda is None:
        raise RuntimeError(f"Activate a compatible {wanted} environment; --no-env-switch uses the current environment")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(conda, [conda, "run", "--no-capture-output", "-n", wanted,
                    "python", "-s", str(Path(__file__).resolve()), *sys.argv[1:], "--no-env-switch"])


"""Embedded by each inference.py; no dependency on the maintenance directory."""
import functools as _ft_functools
import json as _ft_json
import math as _ft_math
import os as _ft_os
import sys as _ft_sys
import tempfile as _ft_tempfile
import time as _ft_time
from contextlib import contextmanager as _ft_contextmanager
from pathlib import Path as _FTPath


class ForwardTiming:
    """Synchronized forward calls only; nested calls are counted once."""

    def __init__(self, device="cpu", sync=None, clock=None):
        self.device = str(device)
        self.seconds = 0.0
        self.depth = 0
        self.active = True
        self._sync = sync
        self._clock = clock or _ft_time.perf_counter
        self._wrapped = []

    def synchronize(self):
        if self._sync is not None:
            self._sync()
        elif self.device.startswith("cuda"):
            torch = _ft_sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

    @_ft_contextmanager
    def forward(self):
        if not self.active or self.depth:
            yield
            return
        self.synchronize()
        start = self._clock()
        self.depth += 1
        try:
            yield
            self.synchronize()
        except BaseException:
            # Failed/OOM attempts do not become successful-sample latency.
            raise
        else:
            self.seconds += self._clock() - start
        finally:
            self.depth -= 1

    def call(self, function, *args, **kwargs):
        with self.forward():
            return function(*args, **kwargs)

    def watch(self, owner, name="forward"):
        original = getattr(owner, name)
        @_ft_functools.wraps(original)
        def measured(*args, **kwargs):
            return self.call(original, *args, **kwargs)
        self._wrapped.append((owner, name, original))
        setattr(owner, name, measured)

    def close(self):
        for owner, name, original in reversed(self._wrapped):
            setattr(owner, name, original)
        self._wrapped.clear()


def _ft_validate(seconds, count):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Timing requires a positive integer sample count")
    seconds = float(seconds)
    if not _ft_math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid forward latency")
    return {"total_latency_s": seconds, "sample_count": count,
            "avg_latency_ms": seconds * 1000.0 / count}


def _ft_atomic(path, value):
    path = _FTPath(path)
    fd, temporary = _ft_tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with _ft_os.fdopen(fd, "w", encoding="utf-8") as stream:
            _ft_json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            _ft_os.fsync(stream.fileno())
        _ft_os.replace(temporary, path)
    finally:
        if _ft_os.path.exists(temporary):
            _ft_os.unlink(temporary)


def save_forward_timing(model_dir, seconds, sample_count):
    model_dir = _FTPath(model_dir).resolve()
    value = _ft_validate(seconds, sample_count)
    PATHS.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    limited = _ft_os.environ.get('MASK_SAMPLE_LIMIT', '0') != '0'
    target = PATHS.RESULT_DIR / ('smoke_inference_time.json' if limited else 'inference_time.json')
    _ft_atomic(target, {model_dir.name: value})
    run = PATHS.RESULT_DIR / 'runs' / _ft_os.environ['MASK_RUN_ID']
    run.mkdir(parents=True, exist_ok=True)
    _ft_atomic(run / 'inference_time.json', {model_dir.name: value})
    print(f'Forward timing saved: {target}', flush=True)
    return value


class ForwardJournal:
    """Per-completed-unit times stay with the external resumable work files."""

    def __init__(self, path):
        self.path = _FTPath(path)
        self.units = _ft_json.loads(self.path.read_text()) if self.path.exists() else {}

    def put(self, key, seconds, count):
        self.units[str(key)] = _ft_validate(seconds, int(count))
        _ft_atomic(self.path, self.units)

    def publish(self, model_dir, *, expected=None):
        count = sum(x["sample_count"] for x in self.units.values())
        if count == 0 or (expected is not None and count != expected):
            print("Forward timing not published: cached predictions have no complete matching timing record.", flush=True)
            return None
        return save_forward_timing(model_dir, sum(x["total_latency_s"] for x in self.units.values()), count)


TIMING_NATIVE_SOURCE = '"""Embedded by each inference.py; no dependency on the maintenance directory."""\nimport functools as _ft_functools\nimport json as _ft_json\nimport math as _ft_math\nimport os as _ft_os\nimport sys as _ft_sys\nimport tempfile as _ft_tempfile\nimport time as _ft_time\nfrom contextlib import contextmanager as _ft_contextmanager\nfrom pathlib import Path as _FTPath\n\n\nclass ForwardTiming:\n    """Synchronized forward calls only; nested calls are counted once."""\n\n    def __init__(self, device="cpu", sync=None, clock=None):\n        self.device = str(device)\n        self.seconds = 0.0\n        self.depth = 0\n        self.active = True\n        self._sync = sync\n        self._clock = clock or _ft_time.perf_counter\n        self._wrapped = []\n\n    def synchronize(self):\n        if self._sync is not None:\n            self._sync()\n        elif self.device.startswith("cuda"):\n            torch = _ft_sys.modules.get("torch")\n            if torch is not None and torch.cuda.is_available():\n                torch.cuda.synchronize(self.device)\n\n    @_ft_contextmanager\n    def forward(self):\n        if not self.active or self.depth:\n            yield\n            return\n        self.synchronize()\n        start = self._clock()\n        self.depth += 1\n        try:\n            yield\n            self.synchronize()\n        except BaseException:\n            # Failed/OOM attempts do not become successful-sample latency.\n            raise\n        else:\n            self.seconds += self._clock() - start\n        finally:\n            self.depth -= 1\n\n    def call(self, function, *args, **kwargs):\n        with self.forward():\n            return function(*args, **kwargs)\n\n    def watch(self, owner, name="forward"):\n        original = getattr(owner, name)\n        @_ft_functools.wraps(original)\n        def measured(*args, **kwargs):\n            return self.call(original, *args, **kwargs)\n        self._wrapped.append((owner, name, original))\n        setattr(owner, name, measured)\n\n    def close(self):\n        for owner, name, original in reversed(self._wrapped):\n            setattr(owner, name, original)\n        self._wrapped.clear()\n\n\ndef _ft_validate(seconds, count):\n    if not isinstance(count, int) or isinstance(count, bool) or count < 1:\n        raise ValueError("Timing requires a positive integer sample count")\n    seconds = float(seconds)\n    if not _ft_math.isfinite(seconds) or seconds < 0:\n        raise ValueError("Invalid forward latency")\n    return {"total_latency_s": seconds, "sample_count": count,\n            "avg_latency_ms": seconds * 1000.0 / count}\n\n\ndef _ft_atomic(path, value):\n    path = _FTPath(path)\n    fd, temporary = _ft_tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)\n    try:\n        with _ft_os.fdopen(fd, "w", encoding="utf-8") as stream:\n            _ft_json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)\n            stream.write("\\n")\n            stream.flush()\n            _ft_os.fsync(stream.fileno())\n        _ft_os.replace(temporary, path)\n    finally:\n        if _ft_os.path.exists(temporary):\n            _ft_os.unlink(temporary)\n\n\ndef save_forward_timing(model_dir, seconds, sample_count):\n    """Merge one model atomically. Lock the existing Model directory: no lock file."""\n    import fcntl\n    model_dir = _FTPath(model_dir).resolve()\n    value = _ft_validate(seconds, sample_count)\n    target = model_dir.parent.parent / "inference_time.json"\n    descriptor = _ft_os.open(model_dir.parent, _ft_os.O_RDONLY)\n    try:\n        fcntl.flock(descriptor, fcntl.LOCK_EX)\n        contents = _ft_json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}\n        if not isinstance(contents, dict):\n            raise ValueError("inference_time.json must be a model-keyed object")\n        contents[model_dir.name] = value\n        _ft_atomic(target, contents)\n    finally:\n        _ft_os.close(descriptor)\n    print(f"Forward timing saved: {target} [{model_dir.name}]", flush=True)\n    return value\n\n\nclass ForwardJournal:\n    """Per-completed-unit times stay with the external resumable work files."""\n\n    def __init__(self, path):\n        self.path = _FTPath(path)\n        self.units = _ft_json.loads(self.path.read_text()) if self.path.exists() else {}\n\n    def put(self, key, seconds, count):\n        self.units[str(key)] = _ft_validate(seconds, int(count))\n        _ft_atomic(self.path, self.units)\n\n    def publish(self, model_dir, *, expected=None):\n        count = sum(x["sample_count"] for x in self.units.values())\n        if count == 0 or (expected is not None and count != expected):\n            print("Forward timing not published: cached predictions have no complete matching timing record.", flush=True)\n            return None\n        return save_forward_timing(model_dir, sum(x["total_latency_s"] for x in self.units.values()), count)\n\n"""Appended to the common helper in the extracted official-runner directory."""\nimport inspect as _native_inspect\n\nTIMER = ForwardTiming("cuda")\nTIMER.active = False\nCOUNT = 0\n\n\ndef begin():\n    global COUNT\n    TIMER.seconds = 0.0\n    TIMER.active = True\n    COUNT = 0\n\n\ndef finish(key, count=None):\n    TIMER.active = False\n    count = COUNT if count is None else int(count)\n    if count > 0:\n        ForwardJournal(_ft_os.environ["FORWARD_JOURNAL"]).put(key, TIMER.seconds, count)\n\n\ndef timed_forward(function):\n    @_ft_functools.wraps(function)\n    def wrapped(*args, **kwargs):\n        return TIMER.call(function, *args, **kwargs)\n    return wrapped\n\n\ndef video_unit(function):\n    @_ft_functools.wraps(function)\n    def wrapped(*args, **kwargs):\n        values = _native_inspect.signature(function).bind(*args, **kwargs).arguments\n        video = values["video_name"]\n        frames = _FTPath(values["base_video_dir"]) / video\n        count = sum(p.suffix.lower() in {".jpg", ".jpeg"} for p in frames.iterdir())\n        begin()\n        try:\n            result = function(*args, **kwargs)\n        except BaseException:\n            TIMER.active = False\n            raise\n        finish("video:" + video, count)\n        return result\n    return wrapped\n\n\ndef gold_forward(model, batch):\n    global COUNT\n    output = TIMER.call(model, batch)\n    # One image-text query, irrespective of number of masks/annotators.\n    COUNT += sum(len(stage.img_ids) for stage in batch.find_inputs)\n    return output\n\n\ndef watch_falcon(engine):\n    # Install AFTER engine construction / CUDA-graph capture. Time graph replay\n    # as well as eager prefill; never instrument the body of a captured graph.\n    TIMER.watch(engine, "_decode_forward")\n    for name in ("forward", "sample_bbox", "get_segm_tokens", "upsample_single_img_features"):\n        TIMER.watch(engine.model, name)\n'

"""Embedded in mask inference.py; modify extracted copies of official code only."""
def _ft_replace(text, before, after, expected=1):
    if text.count(before) != expected:
        raise ValueError(f"Forward timing hook changed: {before[:100]!r}")
    return text.replace(before, after)


def _ft_decorate(text, names, decorator):
    import ast
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    insertions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            first = min([node.lineno] + [x.lineno for x in node.decorator_list]) - 1
            insertions.append((first, " " * node.col_offset + "@_ftn." + decorator + "\n"))
    if not insertions:
        raise ValueError(f"No forward timing boundaries found for {names}")
    for first, value in sorted(insertions, reverse=True):
        lines.insert(first, value)
    text = "".join(lines)
    tree = ast.parse(text)
    last_future = max((n.end_lineno for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"), default=0)
    if not last_future and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        last_future = tree.body[0].end_lineno
    lines = text.splitlines(keepends=True)
    lines.insert(last_future, "\nimport _forward_native as _ftn\n")
    return "".join(lines)


def instrument_native_sources(runtime):
    import ast
    (runtime.root / "_forward_native.py").write_text(TIMING_NATIVE_SOURCE, encoding="utf-8")
    backend = MODEL_SPEC["backend"]
    if backend not in {"video", "sec", "sam31", "gold", "falcon"}:
        return
    for path in runtime.root.rglob("*.py"):
        relative = str(path.relative_to(runtime.root))
        names = None
        if backend in {"video", "sec"} and relative.endswith("/modeling/sam2_base.py"):
            names = {"forward_image", "track_step"}
        elif backend == "sec" and relative.endswith("/inference/modeling_sec.py"):
            names = {"forward"}
        elif backend == "sam31" and "/sam3/model/" in relative and path.name in {
            "sam3_tracker_base.py", "sam3_multiplex_tracking.py", "vl_combiner.py",
            "sam3_image.py", "sam3_multiplex_base.py"}:
            # Direct forward_image/forward_text calls bypass nn.Module.__call__.
            # Nested boundaries share TIMER.depth and are never counted twice.
            tree = ast.parse(path.read_text())
            available = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            names = available & {"forward", "forward_image", "forward_text", "track_step"}
        if names:
            text = _ft_decorate(path.read_text(), names, "timed_forward")
            ast.parse(text)
            path.write_text(text)
    repo = runtime.root / "_shared_test_resources/code" / MODEL_SPEC.get("repo", "")
    if backend in {"video", "sec"}:
        script = repo / ("vos_evaluation/vos_inference.py" if backend == "sec" else "tools/vos_inference.py")
        names = {"vos_separate_inference_per_object"} if backend == "sec" else {"vos_inference", "vos_separate_inference_per_object"}
        script.write_text(_ft_decorate(script.read_text(), names, "video_unit"))
    elif backend == "sam31":
        script = runtime.model / "test_code/vos_point_prompt_eval.py"
        text = script.read_text()
        text = _ft_replace(text, "    for video in videos:\n", "    import _forward_native as _ftn\n    for video in videos:\n        _ftn.begin()\n")
        text = _ft_replace(text, '        finally:\n            predictor.handle_request',
            '            _ftn.finish("video:" + video, len(frame_outputs))\n        finally:\n            predictor.handle_request')
        ast.parse(text)
        script.write_text(text)
    elif backend == "gold":
        script = repo / "sam3/train/trainer.py"
        text = script.read_text()
        text = _ft_replace(text, "        find_stages = model(batch)",
            "        import _forward_native as _ftn\n        find_stages = _ftn.gold_forward(model, batch)")
        text = _ft_replace(text, "        outs = self.val_epoch(dataloader, phase=Phase.VAL)",
            '        import _forward_native as _ftn\n        _ftn.begin()\n'
            '        outs = self.val_epoch(dataloader, phase=Phase.VAL)\n'
            '        _ftn.finish(os.environ["FORWARD_UNIT"])')
        ast.parse(text)
        script.write_text(text)
    elif backend == "falcon":
        script = repo / "eval/pbench.py"
        text = script.read_text()
        text = _ft_replace(text, "    engine.generate(\n",
            "    import _forward_native as _ftn\n    _ftn.watch_falcon(engine)\n    _ftn.begin()\n    engine.generate(\n")
        text = _ft_replace(text, "    wall_time = time.perf_counter() - t0",
            '    _ftn.finish("pbench:level_0", len(sequences))\n    wall_time = time.perf_counter() - t0')
        ast.parse(text)
        script.write_text(text)


class Runtime:
    def __init__(self, args):
        self.args = args
        import torch
        with torch.serialization.safe_globals([bytes]):
            package = torch.load(PATHS.WEIGHTS_PATH, map_location="cpu", weights_only=True, mmap=True)
        weight_identity = hashlib.sha256(json.dumps(package["sha256"], sort_keys=True).encode()).hexdigest()
        del package
        signature = hashlib.sha256(json.dumps({"key": MODEL_SPEC["key"], "location": str(HERE), "weights": weight_identity,
                    "limit": args.max_samples, "device": args.device, "batch_size": args.batch_size,
                    "torch": torch.__version__, "mode": "check" if args.check else "inference",
                    "source": digest(__file__), "metric": digest(PATHS.EVALUATOR)}, sort_keys=True).encode()).hexdigest()[:16]
        self.work = (args.work_dir or (Path(tempfile.gettempdir()) / f"mask-unified-{MODEL_SPEC['key']}-{signature}")).resolve()
        project = PATHS.PROJECT_ROOT
        if self.work == project or self.work.is_relative_to(project):
            raise ValueError("Work directory must be outside the Model/Dataset project")
        self.marker = self.work / ".mask-work.json"
        identity = {"model": MODEL_SPEC["key"], "signature": signature}
        if self.work.exists() and any(self.work.iterdir()):
            if not self.marker.is_file() or json.loads(self.marker.read_text()) != identity:
                raise ValueError("Work directory belongs to another run; choose a separate --work-dir")
        self.work.mkdir(parents=True, exist_ok=True)
        atomic_json(self.marker, identity)
        self.root = self.work / "runtime"
        self.root.mkdir(exist_ok=True)
        self.model = self.root / MODEL_SPEC["key"]
        self.cache = self.work / "cache"
        self.cache.mkdir(exist_ok=True)
        self.result = self.work / "predictions"
        self.result.mkdir(exist_ok=True)
        self.forward_timing = ForwardTiming(args.device)
        self.forward_journal = ForwardJournal(self.work / "forward_time.json")
        self.expected_timing_count = None
        self.timing_complete = True

    def unpack(self):
        os.environ.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "USE_TF": "0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(self.cache / "hf"), "HF_DATASETS_CACHE": str(self.cache / "datasets"),
            "TORCH_HOME": str(self.cache / "torch"), "MPLCONFIGDIR": str(self.cache / "matplotlib")})
        import torch
        torch.set_num_threads(4)
        # Protocol-2 serializes an empty bytes asset using the bytes type.
        # Allow only that inert builtin while retaining weights_only loading.
        with torch.serialization.safe_globals([bytes]):
            package = torch.load(PATHS.WEIGHTS_PATH, map_location="cpu", weights_only=True, mmap=True)
        if package.get("format") != "mask-bundle-v1" or package.get("model_key") != MODEL_SPEC["key"]:
            raise ValueError("Wrong model checkpoint bundle")
        self.metadata = package["metadata"]
        if set(package["files"]) != set(package["sha256"]):
            raise ValueError("Incomplete asset hash manifest")
        for name, payload in package["files"].items():
            path = relative_member(self.root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = package["sha256"][name]
            if path.is_file() and digest(path) == expected:
                continue
            with path.open("wb") as stream:
                if isinstance(payload, bytes):
                    stream.write(payload)
                else:
                    for chunk in payload if isinstance(payload, (list, tuple)) else [payload]:
                        if not isinstance(chunk, torch.Tensor) or chunk.dtype != torch.uint8 or chunk.ndim != 1:
                            raise ValueError(f"Invalid asset payload: {name}")
                        chunk.numpy().tofile(stream)
            if digest(path) != expected:
                raise ValueError(f"Weight/config hash mismatch: {name}")
            path.chmod(package.get("modes", {}).get(name, 0o644))
        for name, source in EMBEDDED_SOURCES.items():
            path = relative_member(self.root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source.replace("__MASK_ROOT__", str(self.root)), encoding="utf-8")
        for name, target in self.metadata.get("links", {}).items():
            path = relative_member(self.root, name)
            destination = relative_member(self.root, target)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_symlink():
                path.symlink_to(os.path.relpath(destination, path.parent), target_is_directory=True)
        self.data = {}
        for name, relative in self.metadata["datasets"].items():
            if Path(relative).is_absolute():
                raise ValueError("Dataset paths must be relative")
            target = PATHS.resolve_legacy_dataset(relative)
            self.data[name] = target
            link = self.root / "_shared_test_resources/data" / name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() and link.resolve() != target:
                link.unlink()
            if not link.is_symlink():
                link.symlink_to(target, target_is_directory=True)
        del package
        if not self.args.check:
            instrument_native_sources(self)
        return self

    def python(self, relative, arguments, repo=None, cwd=None):
        environment = dict(os.environ)
        environment["FORWARD_JOURNAL"] = str(self.forward_journal.path)
        paths = [str(self.root)]
        if repo:
            paths.append(str(self.root / "_shared_test_resources/code" / repo))
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        subprocess.run([sys.executable, "-B", str(self.root / relative), *map(str, arguments)],
                       cwd=str(cwd or self.model), env=environment, check=True)

    def cleanup(self):
        if self.args.keep_work:
            print(f"Work retained: {self.work}")
        elif self.marker.is_file() and self.work.name not in ("", ".", ".."):
            shutil.rmtree(self.work)


def encode_mask(mask):
    import numpy as np
    from pycocotools import mask as masks
    rle = masks.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def completed_records(path):
    """Recover only a truncated final JSONL line; never ignore invalid rows."""
    ids = set()
    if not path.exists():
        return ids
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.seek(start); stream.truncate(); break
            record = json.loads(line)
            key = str(record["sample_id"])
            if key in ids:
                raise ValueError(f"Duplicate cached sample {key}")
            ids.add(key)
    return ids


def point_groups(data, name, maximum):
    """Original LVIS interior points or original SA-1B provided point prompts."""
    count = 0
    if name == "SA-1B-val":
        for file in sorted(data.rglob("sa_*.json")):
            image = file.with_suffix(".jpg")
            if not image.is_file():
                continue
            rows = []
            for index, annotation in enumerate(json.loads(file.read_text()).get("annotations", [])):
                if not annotation.get("point_coords"):
                    continue
                rows.append({"sample_id": f"{file.stem}:{index}", "image": str(image.relative_to(data)),
                             "point": annotation["point_coords"][0],
                             "original": {"annotation_file": str(file.relative_to(data)), "annotation_index": index}})
                count += 1
                if maximum and count >= maximum:
                    break
            if rows:
                yield rows
            if maximum and count >= maximum:
                return
    else:
        import cv2
        import numpy as np
        from urllib.parse import urlparse
        from pycocotools.coco import COCO
        coco = COCO(str(data / "annotations/lvis_v1_val.json"))
        for image_id in sorted(coco.getImgIds()):
            info = coco.imgs[image_id]
            if info.get("coco_url"):
                url = Path(urlparse(info["coco_url"]).path)
                image = Path(url.parent.name) / url.name
            else:
                image = Path("val2017") / info.get("file_name", f"{image_id:012d}.jpg")
            if not (data / image).is_file():
                raise FileNotFoundError(data / image)
            rows = []
            for annotation in coco.imgToAnns[image_id]:
                if annotation.get("iscrowd", False):
                    continue
                truth = coco.annToMask(annotation).astype(np.uint8)
                if not truth.any():
                    continue
                y, x = np.unravel_index(cv2.distanceTransform(truth, cv2.DIST_L2, 5).argmax(), truth.shape)
                rows.append({"sample_id": str(annotation["id"]), "image": str(image), "point": [int(x), int(y)],
                             "original": {"annotation_id": annotation["id"]}})
                count += 1
                if maximum and count >= maximum:
                    break
            if rows:
                yield rows
            if maximum and count >= maximum:
                return


def run_points(runtime, args, metric):
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    name = MODEL_SPEC["dataset"]
    data = runtime.data[name]
    path = runtime.result / "predictions.jsonl"
    done = completed_records(path)
    runtime.expected_timing_count = min(args.max_samples, MODEL_SPEC["count"]) if args.max_samples else MODEL_SPEC["count"]
    expected = min(args.max_samples, MODEL_SPEC["count"]) if args.max_samples else MODEL_SPEC["count"]
    if MODEL_SPEC["backend"].endswith("onnx"):
        manifest = runtime.work / "point_manifest.jsonl"
        with manifest.open("w") as stream:
            count = 0
            for group in point_groups(data, name, args.max_samples):
                for item in group:
                    stream.write(json.dumps(item) + "\n"); count += 1
            if count != expected:
                raise ValueError(f"Prepared dataset has {count}/{expected} instances")
        script = runtime.model / "test_code/inference.mjs"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(NODE_SOURCE)
        node = os.environ.get("MASK_NODE_BINARY") or shutil.which("node")
        if not node:
            raise RuntimeError("Node.js is required for the original ONNX models")
        node_major = int(subprocess.check_output([node, '-p', 'process.versions.node.split(".")[0]'], text=True).strip())
        if node_major < 20:
            raise RuntimeError('Node.js 20+ is required. Activate it with nvm use 20, or set MASK_NODE_BINARY.')
        import site
        libraries = [str(p) for root in site.getsitepackages() for p in (Path(root) / "nvidia").glob("*/lib")]
        libraries.append(str(Path(torch.__file__).parent / "lib"))
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = ":".join(libraries + [environment.get("LD_LIBRARY_PATH", "")])
        subprocess.run([node, str(script), str(runtime.model), str(data), str(manifest), str(path),
                        MODEL_SPEC["backend"], args.device, str(args.batch_size)], env=environment, check=True)
        seconds, measured = 0.0, 0
        for row in metric.jsonl_records(path, expected):
            if "forward_latency_s" in row:
                seconds += row["forward_latency_s"]; measured += 1
        if measured == expected:
            runtime.forward_journal.put("onnx:instances", seconds, measured)
        else:
            runtime.timing_complete = False
    else:
        from transformers import SamModel, SamProcessor
        processor = SamProcessor.from_pretrained(runtime.model, local_files_only=True)
        model = SamModel.from_pretrained(runtime.model, local_files_only=True).to(args.device).eval()
        timer = runtime.forward_timing
        timer.watch(model, "get_image_embeddings")
        timer.watch(model)
        with path.open("a", buffering=1) as stream, torch.inference_mode(), tqdm(total=expected, initial=len(done), unit="instance") as progress:
            for group in point_groups(data, name, args.max_samples):
                pending = [item for item in group if item["sample_id"] not in done]
                if not pending:
                    continue
                with Image.open(data / group[0]["image"]) as image:
                    inputs = processor(image.convert("RGB"), return_tensors="pt")
                timer.seconds = 0.0
                embeddings = model.get_image_embeddings(inputs["pixel_values"].to(args.device))
                encoder_seconds = timer.seconds
                height, width = inputs["original_sizes"][0].tolist()
                new_height, new_width = inputs["reshaped_input_sizes"][0].tolist()
                safe_batch, offset = args.batch_size, 0
                while offset < len(pending):
                    batch = pending[offset:offset + safe_batch]
                    points = torch.tensor([[[[x["point"][0] * new_width / width, x["point"][1] * new_height / height]] for x in batch]], dtype=torch.float32, device=args.device)
                    labels = torch.ones((1, len(batch), 1), dtype=torch.int64, device=args.device)
                    timer.seconds = 0.0
                    try:
                        output = model(image_embeddings=embeddings, input_points=points, input_labels=labels, multimask_output=True)
                        best = output.iou_scores[0].argmax(-1)
                        chosen = output.pred_masks[0, torch.arange(len(batch), device=args.device), best]
                        masks = processor.image_processor.post_process_masks(chosen[None, :, None], inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][:, 0].cpu().numpy()
                    except torch.OutOfMemoryError:
                        if len(batch) == 1:
                            raise
                        safe_batch = max(1, len(batch) // 2)
                        torch.cuda.empty_cache()
                        print(f"Retrying same instances with prompt batch {safe_batch}", flush=True)
                        continue
                    elapsed_forward = timer.seconds + encoder_seconds
                    for item, mask in zip(batch, masks):
                        stream.write(json.dumps({"sample_id": item["sample_id"], "original": item["original"], "prediction": encode_mask(mask), "forward_latency_s": elapsed_forward / len(batch)}) + "\n")
                        done.add(item["sample_id"])
                    stream.flush(); os.fsync(stream.fileno())
                    encoder_seconds = 0.0
                    progress.update(len(batch)); offset += len(batch)
        timer.close()
        del model
        seconds, measured = 0.0, 0
        for row in metric.jsonl_records(path, expected):
            if "forward_latency_s" in row:
                seconds += row["forward_latency_s"]; measured += 1
        if measured == expected:
            runtime.forward_journal.put("pytorch:instances", seconds, measured)
        else:
            runtime.timing_complete = False
    return metric.evaluate(metric.point_records(path, data, name, expected), args.output)


def run_finegrain(runtime, args, metric):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from tqdm.auto import tqdm
    sys.path.insert(0, str(runtime.root / "_shared_test_resources/code/refiners/src"))
    from refiners.solutions import BoxSegmenter
    model = BoxSegmenter(weights=runtime.model / "model.safetensors", device=args.device, margin=0.05)
    timer = runtime.forward_timing
    timer.watch(model.model)
    data = runtime.data[MODEL_SPEC["dataset"]]
    index = 0
    for batch in pq.ParquetFile(data / "data/test-00000-of-00001.parquet").iter_batches(batch_size=8, columns=["image", "mask"]):
        for image_value, mask_value in zip(batch.to_pydict()["image"], batch.to_pydict()["mask"]):
            if args.max_samples and index >= args.max_samples:
                break
            target = runtime.result / f"{index:05d}.png"
            if not target.exists():
                with metric.image_value(image_value) as im:
                    image = im.convert("RGB")
                with metric.image_value(mask_value) as im:
                    truth = np.asarray(im.convert("L")) >= 128
                yy, xx = np.nonzero(truth)
                box = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1) if len(xx) else (0, 0, image.width, image.height)
                timer.seconds = 0.0
                model(image, box_prompt=box).save(target)
                runtime.forward_journal.put("image:" + str(index), timer.seconds, 1)
            elif "image:" + str(index) not in runtime.forward_journal.units:
                runtime.timing_complete = False
            index += 1
            print(f"\rFinegrain {index}/{args.max_samples or 120}", end="", flush=True)
        if args.max_samples and index >= args.max_samples:
            break
    print()
    timer.close()
    runtime.expected_timing_count = index
    return metric.evaluate(metric.finegrain_records(runtime.result, data, args.max_samples), args.output)


def run_medical(runtime, args, metric):
    import numpy as np
    import SimpleITK as sitk
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor
    from tqdm.auto import tqdm
    root = runtime.data["FLARE22Train"] / "FLARE22Train"
    cases = MODEL_SPEC["cases"][:args.max_samples or None]
    processor = SamProcessor.from_pretrained(runtime.model, local_files_only=True)
    model = SamModel.from_pretrained(runtime.model, local_files_only=True).to(args.device).eval()
    timer = runtime.forward_timing
    timer.watch(model)
    with torch.inference_mode():
        for case in cases:
            target = runtime.result / case
            if target.is_file():
                if "case:" + case not in runtime.forward_journal.units:
                    runtime.timing_complete = False
                continue
            timer.seconds = 0.0
            slice_samples = 0
            truth_itk = sitk.ReadImage(str(root / "labels" / case))
            truth = sitk.GetArrayFromImage(truth_itk).astype(np.uint8)
            volume = sitk.GetArrayFromImage(sitk.ReadImage(str(root / "images" / case.replace(".nii.gz", "_0000.nii.gz")))).astype(np.float32)
            pixels = ((np.clip(volume, -160, 240) + 160) / 400 * 255).astype(np.uint8)
            predicted = np.zeros_like(truth)
            for z in tqdm(range(len(truth)), desc=case, unit="slice"):
                image = Image.fromarray(np.repeat(pixels[z, :, :, None], 3, axis=2))
                for label in sorted(int(x) for x in np.unique(truth[z]) if x > 0):
                    yy, xx = np.nonzero(truth[z] == label)
                    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    inputs = processor(image, input_boxes=[[box]], return_tensors="pt")
                    output = model(**{k: v.to(args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}, multimask_output=False)
                    slice_samples += 1
                    mask = processor.image_processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"])[0].reshape(truth[z].shape).numpy().astype(bool)
                    predicted[z][mask] = label
            image = sitk.GetImageFromArray(predicted); image.CopyInformation(truth_itk)
            temporary = target.with_name("partial_" + target.name)
            sitk.WriteImage(image, str(temporary), True); temporary.replace(target)
            runtime.forward_journal.put("case:" + case, timer.seconds, slice_samples)
    timer.close()
    return metric.evaluate(metric.medical_records(runtime.result, runtime.data["FLARE22Train"], cases), args.output)


def run_video(runtime, args, metric):
    if args.device not in ("cuda", "cuda:0"):
        raise ValueError("The official video runners require CUDA; select a GPU with CUDA_VISIBLE_DEVICES and use --device cuda")
    name = MODEL_SPEC["dataset"]
    data = runtime.data[name]
    if name == "DAVIS-2017":
        root = data / "DAVIS"
        frames, annotations = root / "JPEGImages/480p", root / "Annotations/480p"
        videos = (root / "ImageSets/2017/val.txt").read_text().split()
    elif name == "SeCVOS":
        frames, annotations = data / "JPEGImages", data / "Annotations"
        videos = (data / "video_id.txt").read_text().split()
    else:
        root = data / "data/valid"
        frames, annotations = root / "JPEGImages", root / "Annotations"
        videos = sorted(p.name for p in frames.iterdir() if p.is_dir())
    videos = videos[:args.max_samples or None]
    per_object = MODEL_SPEC["backend"] == "sec"
    pending = []
    for video in videos:
        expected = list((annotations / video).glob("*/*.png")) if per_object else list((frames / video).glob("*.jpg"))
        required = [runtime.result / video / p.relative_to(annotations / video) for p in expected] if per_object else [runtime.result / video / (p.stem + ".png") for p in expected]
        if not required or any(not path.is_file() for path in required):
            pending.append(video)
    runtime.expected_timing_count = sum(sum(p.suffix.lower() in {".jpg", ".jpeg"} for p in (frames / video).iterdir()) for video in videos)
    if any("video:" + video not in runtime.forward_journal.units for video in videos if video not in pending):
        runtime.timing_complete = False
    if pending:
        video_list = runtime.work / "videos.txt"
        video_list.write_text("\n".join(pending) + "\n")
        repo = MODEL_SPEC["repo"]
        if MODEL_SPEC["backend"] == "sam31":
            runtime.python(MODEL_SPEC["key"] + "/test_code/vos_point_prompt_eval.py",
                           ["--checkpoint", runtime.model / MODEL_SPEC["checkpoint"], "--frames", frames,
                            "--annotations", annotations, "--video-list", video_list, "--output", runtime.result], repo=repo)
        else:
            command = ["--base_video_dir", frames, "--input_mask_dir", annotations,
                       "--video_list_file", video_list, "--output_mask_dir", runtime.result]
            if per_object:
                command += ["--model_path", runtime.model, "--per_obj_png_file", "--track_object_appearing_later_in_video"]
                script = "vos_evaluation/vos_inference.py"
            else:
                command += ["--sam2_cfg", MODEL_SPEC["config"], "--sam2_checkpoint", runtime.model / MODEL_SPEC["checkpoint"]]
                script = "tools/vos_inference.py"
            runtime.python("_shared_test_resources/code/" + repo + "/" + script, command,
                           repo=repo, cwd=runtime.root / "_shared_test_resources/code" / repo)
    if name == "MOSE-val":
        # First-frame annotations initialize the model; they cannot score VOS.
        import zipfile
        submission = runtime.work / "submission.zip"
        with zipfile.ZipFile(submission, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for video in videos:
                for path in sorted((runtime.result / video).glob("*.png")):
                    archive.write(path, str(path.relative_to(runtime.result)))
        raise RuntimeError(f"Full MOSE GT is server-only. Submit {submission} and use server J; results.json was not overwritten.")
    return metric.evaluate(metric.video_records(runtime.result, annotations, videos, per_object=per_object), args.output)


def run_gold(runtime, args, metric):
    """Keep official Gold full-inference configuration; change only aggregation."""
    repo = runtime.root / "_shared_test_resources/code/sam3"
    data = runtime.data["SACo-Gold"]
    if args.max_samples:
        # Small verification uses the released inference processor on explicit
        # image/query pairs. Its scores are never saved as the full benchmark.
        import numpy as np
        import torch
        from PIL import Image
        from tqdm.auto import tqdm
        sys.path.insert(0, str(repo))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(checkpoint_path=str(runtime.model / "sam3.pt"),
                    bpe_path=str(repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), load_from_HF=False, device=args.device)
        timer = runtime.forward_timing
        timer.watch(model.backbone, "forward_image")
        timer.watch(model.backbone, "forward_text")
        timer.watch(model, "forward_grounding")
        processor = Sam3Processor(model, confidence_threshold=0.3, device=args.device)
        gt = json.loads((data / "gt-annotations/gold_metaclip_merged_a_release_test.json").read_text())
        annotations = defaultdict(list)
        for ann in gt["annotations"]:
            annotations[ann["image_id"]].append(ann["segmentation"])
        records = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            for item in tqdm(gt["images"][:args.max_samples], desc="SAM3 test queries"):
                with Image.open(data / "metaclip-images" / item["file_name"]) as image:
                    state = processor.set_image(image.convert("RGB"))
                output = processor.set_text_prompt(item["text_input"], state=state)
                masks = output["masks"].detach().cpu().numpy()
                pred = [encode_mask(x.reshape(item["height"], item["width"])) for x in masks]
                truth = [metric.rle_on_image_grid(x, item["height"], item["width"]) for x in annotations[item["id"]]]
                records.append({"sample_id": str(item["id"]), "prediction": pred, "truth": truth,
                                "image": data / 'metaclip-images' / item['file_name'],
                                "metadata": {"query": item['text_input'], "subset": "metaclip_nps", "annotator": "a"}})
        runtime.forward_journal.put("gold:limited", timer.seconds, len(records))
        runtime.expected_timing_count = len(records)
        timer.close()
        return metric.evaluate(records, args.output)
    import yaml
    output = runtime.model / "result/gold"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_symlink():
        output.symlink_to(runtime.result, target_is_directory=True)
    for subset in metric.GOLD_SUBSETS:
        prediction = runtime.result / f"gold_{subset}/dumps/gold_{subset}/coco_predictions_segm.json"
        if prediction.exists():
            if "gold:" + subset not in runtime.forward_journal.units:
                runtime.timing_complete = False
            continue
        config_path = repo / "sam3/train/configs/gold_image_evals" / f"sam3_gold_image_{subset}.yaml"
        config = yaml.safe_load(config_path.read_text())
        # The author trainer runs in eval mode and dumps masks. Disable its
        # additional cgF1 calculations: our sibling evaluator computes mIoU.
        for entry in config["trainer"]["meters"]["val"].values():
            entry["cgf1"]["pred_file_evaluators"] = []
        config_path.write_text("# @package _global_\n" + yaml.safe_dump(config, sort_keys=False))
        os.environ["FORWARD_UNIT"] = "gold:" + subset
        runtime.python("_shared_test_resources/code/sam3/sam3/train/train.py",
                       ["-c", f"configs/gold_image_evals/sam3_gold_image_{subset}.yaml",
                        "--use-cluster", "0", "--num-gpus", "1"], repo="sam3", cwd=runtime.model)
    return metric.evaluate(metric.gold_records(runtime.result, data), args.output)


def run_falcon(runtime, args, metric):
    if args.device not in ("cuda", "cuda:0"):
        raise ValueError("The official paged Falcon runner requires CUDA; use CUDA_VISIBLE_DEVICES to choose the GPU")
    repo = runtime.root / "_shared_test_resources/code/Falcon-Perception"
    script = repo / "eval/pbench.py"
    records = runtime.result / "predictions.jsonl"
    completion = runtime.result / "inference_complete.json"
    if not completion.exists():
        text = script.read_text()
        anchor = "        per_sample.append(metrics.sample_f1(pred_rles, gt_rles, iou_thresholds))"
        capture = '''        capture_path = os.environ.get("MASK_CAPTURE_PATH")
        if capture_path:
            from pathlib import Path as CapturePath
            original_image = sample["image"]
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            original_image.save(CapturePath(capture_path).parent / f"source_{seq.request_idx}.png")
            with open(capture_path, "a", encoding="utf-8") as capture:
                capture.write(json.dumps({"sample_id": str(seq.request_idx), "prediction": pred_rles, "truth": gt_rles, "source_image": f"source_{seq.request_idx}.png", "metadata": {"query": sample["expression"], "split": "level_0"}}, default=lambda x: x.decode("ascii") if isinstance(x, bytes) else x) + "\\n")
'''
        if anchor not in text:
            raise ValueError("Upstream PBench scoring hook changed")
        script.write_text(text.replace(anchor, capture + anchor))
        # Falcon generates its batch jointly; incomplete prior predictions are
        # never mixed with a fresh run. Only this run-owned temporary file resets.
        records.write_text("")
        previous = os.environ.get("MASK_CAPTURE_PATH")
        os.environ["MASK_CAPTURE_PATH"] = str(records)
        try:
            runtime.python("_shared_test_resources/code/Falcon-Perception/eval/pbench.py",
                           ["--hf-local-dir", runtime.model, "--dataset", runtime.data["PBench"], "--split", "level_0",
                            "--limit", str(args.max_samples), "--device", args.device, "--dtype", "bfloat16",
                            "--max-dimension", "1024", "--out-dir", runtime.work / "native-diagnostics"],
                           repo="Falcon-Perception", cwd=repo)
        finally:
            if previous is None:
                os.environ.pop("MASK_CAPTURE_PATH", None)
            else:
                os.environ["MASK_CAPTURE_PATH"] = previous
        expected = min(args.max_samples, 1930) if args.max_samples else 1930
        actual = sum(1 for _ in metric.jsonl_records(records, expected))
        atomic_json(completion, {"samples": actual})
    runtime.expected_timing_count = min(args.max_samples, 1930) if args.max_samples else 1930
    expected = min(args.max_samples, 1930) if args.max_samples else 1930
    return metric.evaluate(metric.captured_falcon_records(records, expected), args.output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=MODEL_SPEC.get("batch_size", 16))
    parser.add_argument("--max-samples", "--max-images", dest="max_samples", type=int, default=0, help="instances, images, cases, or complete videos depending on the model")
    parser.add_argument("--output", type=Path, default=PATHS.RESULT_DIR / "results.json")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-env-switch", action="store_true", help="Use current Python instead of named conda environment")
    args = parser.parse_args()
    args.output = PATHS.project_path(args.output)
    if args.work_dir is not None:
        args.work_dir = PATHS.project_path(args.work_dir)
    os.environ.setdefault('MASK_RUN_ID', uuid.uuid4().hex)
    os.environ['MASK_SAMPLE_LIMIT'] = str(args.max_samples)
    if args.batch_size < 1 or args.max_samples < 0:
        parser.error("Invalid batch size/sample limit")
    if args.max_samples and args.output == PATHS.RESULT_DIR / "results.json":
        args.output = PATHS.RESULT_DIR / 'smoke_results.json'
    if args.output.is_relative_to(PATHS.DATASET_ROOT):
        parser.error('Metric output must not overwrite dataset files')
    maybe_switch_environment(args)
    runtime = Runtime(args).unpack()
    if args.check:
        print(json.dumps({"model": MODEL_SPEC["key"], "weights": "hash-verified", "datasets": runtime.metadata["datasets"]}))
        runtime.cleanup()
        return
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    metric = load_metric()
    evaluate_metric = metric.evaluate
    def evaluate_after_forward(*values, **options):
        # Publish immediately after inference, before CPU metrics / result I/O.
        if runtime.timing_complete:
            journal = ForwardJournal(runtime.forward_journal.path)
            journal.publish(HERE, expected=runtime.expected_timing_count)
        coco_spec = importlib.util.spec_from_file_location('_model_coco_export', Path(__file__).with_name('bundle.py'))
        coco = importlib.util.module_from_spec(coco_spec)
        coco_spec.loader.exec_module(coco)
        writer = coco.BundleWriter(HERE, 'mask_generation', {
            'dataset': MODEL_SPEC['dataset'], 'backend': MODEL_SPEC['backend'],
            'sample_limit': args.max_samples,
            'batch_policy': 'native-runner-default' if MODEL_SPEC['backend'] not in ('sam_hf', 'slim_onnx', 'sam3_onnx') else 'project-prompt-batch',
            'configured_prompt_batch_size': args.batch_size if MODEL_SPEC['backend'] in ('sam_hf', 'slim_onnx', 'sam3_onnx') else None,
            'metric': 'miou', 'aggregation': 'existing-evaluate-record-groups',
            'matching': 'paired-or-Hungarian-by-unit-kind', 'both_empty_iou': 1.0})
        for record in values[0]:
            writer.mask(record)
        protocol_path = writer.finish()
        output = values[1] if len(values) > 1 else options.get('output_path')
        # Call the original metric via the standalone bundle reader; never
        # recurse into this export hook or require the transient runtime again.
        return metric.evaluate_coco(protocol_path.parent, output)
    metric.evaluate = evaluate_after_forward
    backend = MODEL_SPEC["backend"]
    runners = {"sam_hf": run_points, "slim_onnx": run_points, "sam3_onnx": run_points,
               "finegrain": run_finegrain, "medical": run_medical, "video": run_video,
               "sam31": run_video, "sec": run_video, "gold": run_gold, "falcon": run_falcon}
    try:
        result = runners[backend](runtime, args, metric)
    except BaseException:
        print(f"Incomplete run; resume files retained at {runtime.work}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False))
    print(f"Saved: {args.output}")
    runtime.cleanup()


if __name__ == "__main__":
    main()
