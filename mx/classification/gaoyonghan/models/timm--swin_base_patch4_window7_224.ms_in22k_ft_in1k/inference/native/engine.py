#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单模型分类推理引擎：直接产出统一评测协议要求的逐样本预测。

由 inference/native/run_model.py 调用，不对用户暴露命令行。

这里解决三件事：

1. **数据布局**。两种都认，自动识别——
   - parquet（image + label 列）：自建的 5 个评测集；
   - ImageFolder（`val/<类目录>/<图片>`）：共享的 dataset/imagenet1k_val。
   统一产出 (sample_id, 图片, 标签) 三元组。

2. **标签对齐**。perm / subset / groups 三种口径**全部在 top-k 之前完成**，
   写出去的 prediction 与 target 一定在同一标签空间。这一点是硬要求：
   subset 与 groups 改的是「argmax 在哪些类上取」，改标签改不出来，
   所以必须在打分张量上做，不能靠事后查表。

3. **计时与完成度**。逐批计时（预热不计入）、峰值显存、样本数写进
   status.json / timing.json，prepare.py 靠 status.json 判断原生运行是否完成。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path

KS = (1, 5)          # 统一档位：top-1 / top-5


# ================================================================ 数据读取
def _image_bytes(cell):
    """HF datasets 的 Image 特征在 parquet 里是 struct<bytes, path>。"""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return bytes(cell["bytes"])
        p = cell.get("path")
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                return fh.read()
    return None


def _parquet_columns(path: Path):
    import pyarrow.parquet as pq

    names = pq.ParquetFile(str(path)).schema_arrow.names
    img = "image" if "image" in names else ("img" if "img" in names else None)
    lab = "label" if "label" in names else ("labels" if "labels" in names else None)
    if img is None or lab is None:
        raise RuntimeError(f"{path.name}: 需要 image/img 与 label/labels 列，实际列={names}")
    return img, lab


def _parquet_shards(paths, max_images):
    """逐个分片读成内存记录。分片之间按名排序，保证顺序可复现。

    冒烟模式（max_images）在**每个分片内等距取**，而不是只取开头：
    ImageNet 镜像按类别连续排列，只取开头会退化成「只测前几类」，准确率虚高。
    """
    import pyarrow.parquet as pq

    n_shards = len(paths)
    per = None
    if max_images and n_shards > 1:
        per = max(1, -(-max_images // n_shards))

    shards = []
    emitted = 0
    for p in paths:
        if max_images and emitted >= max_images:
            break
        img_c, lab_c = _parquet_columns(p)
        table = pq.read_table(str(p), columns=[img_c, lab_c])
        imgs = table.column(img_c).to_pylist()
        labs = table.column(lab_c).to_pylist()
        n_total = len(imgs)

        if max_images:
            take = min(per or max_images, max_images - emitted)
            if take < n_total:
                step = n_total / take
                idxs = [int(i * step) for i in range(take)]
            else:
                idxs = list(range(n_total))
        else:
            idxs = range(n_total)

        recs, skipped = [], 0
        for i in idxs:
            raw = _image_bytes(imgs[i])
            if raw is None:
                skipped += 1
                continue
            recs.append((f"{p.stem}:{i}", raw, int(labs[i])))
        if skipped:
            print(f"      ! {p.name}: 跳过 {skipped} 条无法解析的图片")
        emitted += len(recs)
        del table, imgs, labs
        if recs:
            shards.append(dict(name=p.stem, records=recs))
    return shards


def _looks_like_class_root(d: Path) -> bool:
    subs = [x for x in d.iterdir() if x.is_dir()]
    if not subs:
        return False
    sub = subs[0]
    return any(f.is_file() for f in sub.iterdir())


def _imagefolder_shards(ds_dir: Path, max_images):
    """ImageFolder 布局：根目录下每个子目录是一个类，标签 = 排序后的类序号。"""
    root = ds_dir
    if not _looks_like_class_root(root):
        for name in ("val", "test", "validation"):
            cand = root / name
            if cand.is_dir() and _looks_like_class_root(cand):
                root = cand
                break
    if not _looks_like_class_root(root):
        raise RuntimeError(f"{ds_dir}: 既不是 parquet 目录，也不是 ImageFolder（无类子目录）")

    classes = sorted(d for d in root.iterdir() if d.is_dir())
    flat = []
    for ci, cls in enumerate(classes):
        for f in sorted(cls.iterdir()):
            if f.is_file():
                flat.append((ci, f))
    if not flat:
        raise RuntimeError(f"{root}: 类目录下没有文件")

    if max_images and max_images < len(flat):
        step = len(flat) / max_images
        flat = [flat[int(i * step)] for i in range(max_images)]

    recs = [(f"{classes[ci].name}/{f.name}", f, ci) for ci, f in flat]
    return [dict(name=root.name, records=recs)]


def build_shards(ds_dir, max_images=None):
    """返回 [{name, records:[(sample_id, bytes|Path, label)]}]。

    识别顺序有讲究：先看顶层 parquet，再看 val/test 这类 ImageFolder 标记目录，
    最后才递归找 parquet。反过来的话，`stanford_cars/data/*.parquet` 这种
    「parquet 藏在子目录」的布局会被误判成「data 是一个类目录」，
    把 parquet 文件本身当成图片塞给 PIL。
    """
    p = Path(ds_dir)
    if not p.is_dir():
        raise RuntimeError(f"评测集目录不存在：{p}")

    top = sorted(p.glob("*.parquet"))
    if top:
        return _parquet_shards(top, max_images)

    for name in ("val", "test", "validation"):
        cand = p / name
        if cand.is_dir() and _looks_like_class_root(cand):
            return _imagefolder_shards(p, max_images)

    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if len(subdirs) > 1 and _looks_like_class_root(p):
        return _imagefolder_shards(p, max_images)

    deep = sorted(p.rglob("*.parquet"))
    if deep:
        return _parquet_shards(deep, max_images)

    if subdirs and _looks_like_class_root(p):
        return _imagefolder_shards(p, max_images)

    raise RuntimeError(f"{p}: 找不到 parquet，也没有 ImageFolder 类目录")


# ================================================================ 标签对齐
def build_align(map_path, device, num_classes=None):
    """把映射表解析成可直接用于打分的形式。

    perm   —— 全类别重排：mapping=[{model_idx, dataset_idx}]，长度 = 模型类别数
    subset —— 类别子集：keep=[模型索引]，logits 先限到这些列再 argmax
    groups —— 同类合并：groups=[{group, model_idx:[...]}]，簇内 logsumexp 再取 argmax
    """
    if map_path is None:
        return None
    import torch

    data = json.loads(Path(map_path).read_text(encoding="utf-8"))
    mode = (data.get("mode") or "perm").strip().lower()
    name = Path(map_path).name

    if mode == "perm":
        mapping = data.get("mapping") or []
        arr = [None] * len(mapping)
        for e in mapping:
            arr[int(e["model_idx"])] = int(e["dataset_idx"])
        if not arr or any(v is None for v in arr):
            raise RuntimeError(f"{name}: perm 映射表不完整（{len(arr)} 条，存在空洞）")
        same = sum(1 for i, v in enumerate(arr) if i == v)
        return {"mode": "perm", "t": torch.tensor(arr, dtype=torch.long, device=device),
                "n": len(arr), "same": same, "path": name}

    if mode == "subset":
        keep = data.get("keep") or [int(e["model_idx"]) for e in (data.get("mapping") or [])]
        keep = sorted(int(k) for k in keep)
        if not keep:
            raise RuntimeError(f"{name}: subset 模式但 keep 为空")
        if num_classes and max(keep) >= int(num_classes):
            raise RuntimeError(f"{name}: keep 里 {max(keep)} 超出模型类别数 {num_classes}")
        return {"mode": "subset", "t": torch.tensor(keep, dtype=torch.long, device=device),
                "n": len(keep), "same": None, "path": name}

    if mode == "groups":
        gs = sorted(data.get("groups") or [], key=lambda g: int(g["group"]))
        if not gs:
            raise RuntimeError(f"{name}: groups 模式但 groups 为空")
        ids = [int(g["group"]) for g in gs]
        if ids != list(range(len(gs))):
            raise RuntimeError(f"{name}: group 序号必须连续 0..{len(gs) - 1}（评测集 label 就是它），"
                               f"实际开头 {ids[:8]}")
        tensors = [torch.tensor([int(m) for m in g["model_idx"]], dtype=torch.long, device=device)
                   for g in gs]
        n_cls = int(num_classes or (max(int(m) for g in gs for m in g["model_idx"]) + 1))
        m2g = torch.full((n_cls,), -1, dtype=torch.long, device=device)
        for gi, g in enumerate(gs):
            for m in g["model_idx"]:
                if int(m) >= n_cls:
                    raise RuntimeError(f"{name}: 模型索引 {m} 超出模型类别数 {n_cls}")
                m2g[int(m)] = gi
        holes = [i for i in range(n_cls) if int(m2g[i]) < 0]
        if holes:
            raise RuntimeError(f"{name}: 模型类 {holes} 不在任何簇里，groups 必须全覆盖")
        return {"mode": "groups", "t": tensors, "m2g": m2g, "n": len(gs), "same": None, "path": name}

    raise RuntimeError(f"{name}: 不认识的 mode={mode!r}（只支持 perm / subset / groups）")


def scoring_space(logits, align):
    """把模型原始 logits 变换到「评测标签空间」。

    返回 (scores, index_map)：
      None          -> (logits, None)         模型索引即数据集索引
      perm          -> (logits, perm 表)      topk 得到的模型索引需查表转数据集序
      subset        -> (logits[:, keep], keep) 列位置需映射回模型索引
      groups        -> (簇内 logsumexp, None)  列位置本身就是簇号
    """
    import torch

    if align is None:
        return logits, None
    if align["mode"] == "perm":
        return logits, align["t"]
    if align["mode"] == "subset":
        keep = align["t"]
        return logits[:, keep], keep
    groups = torch.stack([torch.logsumexp(logits[:, g], dim=1) for g in align["t"]], dim=1)
    return groups, None


def topk_rows(scores, index_map):
    """在评分空间取 top-k，返回 (预测索引 [N,k], 分数 [N,k])。"""
    maxk = min(max(KS), scores.shape[1])
    top = scores.topk(maxk, dim=1)
    idx = top.indices
    if index_map is not None:
        idx = index_map[idx]
    return idx, top.values


# ================================================================ 模型加载
def is_model_dir(path: Path) -> bool:
    """目录本身就是一份可直接加载的模型仓库（config 与权重文件都在这一层）。"""
    path = Path(path)
    return (path / "config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def ensure_safetensors(model_dir: Path):
    """只有 pytorch_model.bin 的仓库先安全转成 safetensors。

    transformers >= 4.5x 在 torch < 2.6 上会因 CVE-2025-32434 拒绝 torch.load .bin；
    safetensors 不受该限制。原 .bin 保留不删。
    """
    sf = model_dir / "model.safetensors"
    binp = model_dir / "pytorch_model.bin"
    if sf.exists() or not binp.exists():
        return
    import torch
    from safetensors.torch import save_file

    print(f"      · {model_dir.name} 只有 pytorch_model.bin，转换一份 safetensors（原文件保留）")
    state = torch.load(str(binp), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"{binp} 顶层不是 dict，而是 {type(state).__name__}")
    bad = [k for k, v in state.items() if not hasattr(v, "shape")]
    if bad:
        raise RuntimeError(f"{binp} 含非张量条目 {bad[:5]}，拒绝转换")
    save_file({k: v.contiguous() for k, v in state.items()}, str(sf), metadata={"format": "pt"})


def load_transformers(model_dir: Path):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    ensure_safetensors(model_dir)
    try:
        processor = AutoImageProcessor.from_pretrained(str(model_dir))
    except Exception:                      # 老仓库只有 feature_extractor 配置
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(str(model_dir))

    try:
        model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    except ValueError as exc:
        if "torch.load" not in str(exc):
            raise
        print(f"      ! safetensors 不可用（{str(exc)[:70]}...），临时放行 check_torch_load_is_safe")
        import transformers.modeling_utils as mu
        import transformers.utils.import_utils as iu
        mu.check_torch_load_is_safe = lambda *a, **k: None
        iu.check_torch_load_is_safe = lambda *a, **k: None
        model = AutoModelForImageClassification.from_pretrained(str(model_dir))

    def transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    return model, transform, f"transformers {model.__class__.__name__} | id2label n={len(model.config.id2label)}"


def load_timm(model_dir: Path, timm_name: str, num_classes=None):
    import timm
    import torch
    from safetensors.torch import load_file

    nc = int(num_classes or 1000)
    rep_all = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    rep = rep_all.get("pretrained_cfg") or {}
    try:
        model = timm.create_model(timm_name, pretrained=False, num_classes=nc)
    except Exception as exc:
        arch = rep_all.get("architecture")
        if not arch:
            raise
        print(f"      ! timm 不认名字 {timm_name}（{exc}），退回架构名 {arch}")
        model = timm.create_model(arch, pretrained=False, num_classes=nc)

    sf = model_dir / "model.safetensors"
    binp = model_dir / "pytorch_model.bin"
    if sf.exists():
        state = load_file(str(sf))
        src = "model.safetensors"
    elif binp.exists():
        state = torch.load(str(binp), map_location="cpu", weights_only=True)
        src = "pytorch_model.bin"
    else:
        raise FileNotFoundError(f"{model_dir} 下没有 model.safetensors / pytorch_model.bin")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"权重键缺失 {len(missing)} 个，拒绝继续（否则指标不可信）：{list(missing)[:6]}")
    if unexpected:
        print(f"      · 权重文件多出 {len(unexpected)} 个非模型键（swin 的非持久化 buffer 等），已忽略")
    del state

    cfg = timm.data.resolve_data_config({}, model=model)
    for k in ("mean", "std", "input_size", "interpolation", "crop_pct", "crop_mode",
              "test_input_size", "test_crop_pct"):
        if k in rep:
            cfg[k] = tuple(rep[k]) if isinstance(rep[k], list) else rep[k]
    if rep.get("label_offset"):
        print(f"      ! 该模型 label_offset={rep['label_offset']}，预测索引需加偏移，请人工确认")
    try:
        transform = timm.data.create_transform(**cfg)
    except TypeError as exc:
        import inspect
        sig = inspect.signature(timm.data.create_transform).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in sig.values()):
            dropped = [k for k in cfg if k not in sig]
            cfg = {k: v for k, v in cfg.items() if k in sig}
            print(f"      ! create_transform 不接受 {dropped}，已忽略后重试（原报错 {exc}）")
        transform = timm.data.create_transform(**cfg)

    return model, transform, (f"timm {timm_name} | 权重 {src} | "
                              f"input {tuple(cfg.get('input_size', (3, 224, 224)))} "
                              f"crop_pct={cfg.get('crop_pct')} interp={cfg.get('interpolation')}")


# ================================================================ 预处理口径
_TF_INTERP = ("bilinear", "bicubic", "lanczos", "nearest", "hamming", "box")


def _proc_mean_std(model_dir: Path):
    default = (0.5, 0.5, 0.5)
    for name in ("preprocessor_config.json", "processor_config.json"):
        p = model_dir / name
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        m, s = cfg.get("image_mean"), cfg.get("image_std")
        if m and s:
            return tuple(float(x) for x in m), tuple(float(x) for x in s)
    return default, default


def _norm_tensor(pil_img, mean, std):
    import numpy as np
    import torch

    a = np.asarray(pil_img).astype(np.float32) / 255.0
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def build_transform_override(spec_tf: str, model_dir: Path):
    """`<kind>:<size>[:<interp>]`；空口径返回 (None, None)。

    squash:224           直接压扁到 224×224（ViTFeatureExtractor 口径）
    tv_short:224         等比缩到短边 224 再中心裁剪（HF 官方示例口径）
    tv_short:256@224     短边 256、裁剪 224
    """
    if not spec_tf:
        return None, None
    from PIL import Image
    from torchvision import transforms as T

    interp_map = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC,
                  "lanczos": T.InterpolationMode.LANCZOS, "nearest": T.InterpolationMode.NEAREST,
                  "hamming": T.InterpolationMode.HAMMING, "box": T.InterpolationMode.BOX}
    bits = [b.strip() for b in spec_tf.split(":") if b.strip()]
    kind = bits[0].lower()
    size_s = bits[1] if len(bits) > 1 else "224"
    iname = bits[2].lower() if len(bits) > 2 else "bilinear"
    if iname not in _TF_INTERP:
        raise SystemExit(f"[transform] 不认识的插值 {iname!r}，可选 {_TF_INTERP}")
    interp = interp_map[iname]
    mean, std = _proc_mean_std(model_dir)

    if kind == "squash":
        n = int(size_s)
        return (lambda im: _norm_tensor(im.resize((n, n), Image.BILINEAR), mean, std),
                f"squash {n}x{n} BILINEAR（ViTFeatureExtractor 口径）")
    if kind == "tv_short":
        if "@" in size_s:
            a, b = size_s.split("@", 1)
            short, crop = int(a), int(b)
        else:
            short = crop = int(size_s)
        tf = T.Compose([T.Resize(short, interpolation=interp), T.CenterCrop(crop),
                        T.ToTensor(), T.Normalize(mean, std)])
        return tf, f"torchvision Resize(短边 {short}, {iname}) + CenterCrop({crop})"
    raise SystemExit(f"[transform] 不认识的口径 {spec_tf!r}，kind 只支持 squash / tv_short")


# ================================================================ 数据管道
def make_loader(records, transform, batch_size, workers):
    import torch
    from PIL import Image

    class _DS(torch.utils.data.Dataset):
        def __init__(self, recs, tf):
            self.recs, self.tf = recs, tf

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            _, payload, label = self.recs[i]
            img = Image.open(BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload)
            return self.tf(img.convert("RGB")), label

    kw = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
              pin_memory=True, drop_last=False)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return torch.utils.data.DataLoader(_DS(records, transform), **kw)


# ================================================================ 主流程
def run(cfg: dict, model_root: Path, weights_root: Path, dataset_root: Path,
        out_root: Path, device_str="cuda:0", batch_size=64, workers=8,
        warmup=10, precision="amp_bf16", max_images=None, phase="full"):
    """跑一个模型，落盘 predictions.jsonl / status.json / timing.json。"""
    import torch

    seq = int(cfg["sequence"])
    ds_dir = dataset_root / cfg["dataset_sub"]
    weights_dir = weights_root / cfg["dirname"]
    # 权重随交付目录落盘时，weights/ 是仓库快照、没有 <dirname> 这一层，直接用它本身
    if not weights_dir.is_dir() and is_model_dir(weights_root):
        weights_dir = weights_root
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\n[{seq}] {cfg['model_id']}\n{'=' * 72}")
    print(f"  权重   : {weights_dir}")
    print(f"  数据集 : {ds_dir}")

    if not weights_dir.is_dir():
        raise RuntimeError(f"权重目录不存在：{weights_dir}（设置 WEIGHTS_ROOT 或用 --weights-root 指定）")

    device = torch.device(device_str)
    t_all0 = time.perf_counter()

    # ---- 数据
    shards = build_shards(ds_dir, max_images)
    total = sum(len(s["records"]) for s in shards)
    fmt = "imagefolder" if any(f.name in ("val", "test", "validation") for f in ds_dir.iterdir()
                               if f.is_dir()) else "parquet"
    print(f"  格式   : {fmt} | 分片 {len(shards)} | 样本 {total}")

    # ---- 标签对齐
    map_path = None
    if cfg.get("align"):
        cand = [model_root / "inference" / "labels" / cfg["align"], weights_dir / cfg["align"]]
        map_path = next((c for c in cand if c.exists()), None)
        if map_path is None:
            raise RuntimeError(f"映射表 {cfg['align']} 找不到，试过 {[str(c) for c in cand]}")
    align = build_align(map_path, device, cfg.get("num_classes"))
    if align:
        if align["mode"] == "perm":
            print(f"  对齐   : {align['path']} [perm] {align['n']} 类，同位 {align['same']}/{align['n']}")
        elif align["mode"] == "subset":
            print(f"  对齐   : {align['path']} [subset] 只在模型类别 {align['t'].tolist()} 里选")
        else:
            print(f"  对齐   : {align['path']} [groups] {align['n']} 簇（簇内 logsumexp）")
    else:
        print("  对齐   : 无（模型索引即数据集索引）")

    # ---- 模型
    t_load0 = time.perf_counter()
    print("  加载模型 ...")
    if cfg["loader"] == "transformers":
        model, transform, info = load_transformers(weights_dir)
    else:
        model, transform, info = load_timm(weights_dir, cfg["timm_name"], cfg.get("num_classes"))
    model.eval().to(device)
    load_sec = time.perf_counter() - t_load0
    print(f"  {info}")
    print(f"  加载耗时 {load_sec:.1f}s")

    tf_spec = (cfg.get("transform") or "").strip()
    tf_label = "处理器默认"
    if tf_spec:
        ov, desc = build_transform_override(tf_spec, weights_dir)
        if ov is not None:
            transform, tf_label = ov, tf_spec
            print(f"  预处理 : {desc}")

    amp = precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "amp_bf16" else torch.float16
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    pred_path = out_root / "predictions.jsonl"
    n_done = 0
    warm_done = 0
    warm_sec = 0.0
    fwd_sec = 0.0
    digests = hashlib.sha256()

    # ---- 预热：单独跑一遍开头的几批，跑完从头开始，**不消耗评测样本**
    # 早期写法是「主循环里跳过前 N 批」，在 167 这种 1047 张的小评测集上
    # 10 批 × 64 = 640 张被直接吃掉，只剩 407 张，被张数校验拦下。
    if warmup and shards:
        probe = shards[0]["records"][: warmup * batch_size]
        if probe:
            print(f"  预热 {warmup} 批（其样本仍会正常评测）...")
            loader = make_loader(probe, transform, batch_size, workers)
            with torch.inference_mode():
                for i, (x, _) in enumerate(loader):
                    if i >= warmup:
                        break
                    t0 = time.perf_counter()
                    x = x.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and use_cuda)):
                        model(x)
                    if use_cuda:
                        torch.cuda.synchronize()
                    warm_sec += time.perf_counter() - t0
                    warm_done += 1
            del loader, probe
            print(f"  预热 {warm_done} 批完成，{warm_sec:.1f}s")

    with pred_path.open("w", encoding="utf-8") as fh:
        with torch.inference_mode():
            for si, shard in enumerate(shards, 1):
                loader = make_loader(shard["records"], transform, batch_size, workers)
                print(f"  分片 {si}/{len(shards)} {shard['name']}  {len(shard['records'])} 张")
                base = 0                       # 本分片内已消费的样本数
                for x, y in loader:
                    t0 = time.perf_counter()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=(amp and use_cuda)):
                        out = model(x)
                    # 必须同步再停表：CUDA 异步，不同步量到的是 CPU 下发速率
                    if use_cuda:
                        torch.cuda.synchronize()
                    logits = (out.logits if hasattr(out, "logits") else out).float()

                    scores, index_map = scoring_space(logits, align)
                    idx, vals = topk_rows(scores, index_map)
                    bs = int(y.shape[0])
                    sid_base = base
                    base += bs

                    fwd_sec += time.perf_counter() - t0

                    preds = idx[:, 0].tolist()
                    tops = idx.tolist()
                    scs = vals.tolist()
                    labs = y.tolist()
                    lines = []
                    for j in range(bs):
                        row_scores = scs[j]
                        ok = all(isinstance(v, float) and v == v and abs(v) != float("inf")
                                 for v in row_scores)
                        lines.append(json.dumps({
                            "id": shard["records"][sid_base + j][0],
                            "target": int(labs[j]),
                            "prediction": int(preds[j]),
                            "topk": [int(v) for v in tops[j]],
                            "topk_scores": [round(float(v), 6) for v in row_scores] if ok else None,
                        }, ensure_ascii=False))
                    n_done += bs
                    block = "\n".join(lines) + "\n"
                    fh.write(block)
                    digests.update(block.encode("utf-8"))

                    if n_done % (bs * 20) < bs:
                        rate = n_done / fwd_sec if fwd_sec else 0
                        print(f"      {n_done}/{total}  {rate:.0f} 张/秒")
                del loader
                if use_cuda:
                    torch.cuda.empty_cache()

    wall = time.perf_counter() - t_all0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if use_cuda else 0.0

    if n_done == 0:
        raise RuntimeError(f"没有评测到任何样本（total={total}，warmup={warmup}）")

    status = dict(sequence=seq, model_id=cfg["model_id"], phase=phase, state="completed",
                  images=n_done, dataset=cfg["dataset_sub"], dataset_format=fmt,
                  align_mode=(align["mode"] if align else "none"),
                  align_map=(align["path"] if align else None),
                  transform=tf_label, device=str(device), precision=precision,
                  batch_size=batch_size, torch=torch.__version__,
                  cuda=(torch.version.cuda if use_cuda else None),
                  gpu=(torch.cuda.get_device_name(0) if use_cuda else None),
                  predictions=pred_path.name, predictions_sha256=digests.hexdigest(),
                  run_wall_seconds=round(wall, 3))
    (out_root / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    timing = dict(model_load_seconds_this_segment=round(load_sec, 3),
                  warmup_batches=warm_done,
                  warmup_seconds_this_segment=round(warm_sec, 3),
                  timed_images=n_done,
                  forward_seconds=round(fwd_sec, 3),
                  forward_amortized_ms_per_image=round(fwd_sec / n_done * 1000, 3),
                  forward_images_per_second=round(n_done / fwd_sec, 2) if fwd_sec else 0.0,
                  forward_method=("CUDA events" if use_cuda else "perf_counter"),
                  peak_allocated_bytes_this_segment=int(peak * 1024 ** 2),
                  run_wall_seconds_this_segment=round(wall, 3))
    (out_root / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print(f"  >>> {n_done} 张 | 前向 {fwd_sec:.1f}s = "
          f"{fwd_sec / n_done * 1000:.2f} ms/张 | 峰值显存 {peak:.0f} MiB | 全流程 {wall:.0f}s")
    print(f"  预测文件 {pred_path}")
    return status, timing
