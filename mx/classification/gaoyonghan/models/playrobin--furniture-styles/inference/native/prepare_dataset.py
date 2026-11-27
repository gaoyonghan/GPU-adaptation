#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
167 / 168 / 170 —— 评测集预处理（统一产出 parquet，供 run_inference.py 直接吃）

设计原则：**引擎零改动**。run_inference.py 只认 parquet 的 `image` + `label/labels`，
所以把「筛类、换标签空间、SD 生成、zip 选择性解压」这些脏活全收在这个脚本里。

三行的原始素材形态完全不同：
  167  tonijhanel/my_interior_design_dataset（HF parquet，11 个 style 要筛 3 个）
       -> 取官方 test split，筛 {Bohemian, Coastal, Mid-Century Modern}，
          标签换到【模型索引空间】= {1, 2, 4}
  168  作者用 SD2.1 现场生成（无公开数据）
       -> 用 SD2.1 base 按作者机制重建：同一 prompt/seed 下 30 步跑完出 clean，
          30 步调度【只跑前 10 步就解码】出「半成品」(unfinished passes) 作 blurry
          另出一份「clean 图 + 高斯模糊」作对照，用来证明模型学的是 SD 伪影而非通用模糊
  170  Indian Medicinal Plants（13.28GB med.zip）
       -> 读 zip 中央目录拿类文件夹名 -> 归一化匹配模型的 52 类 -> 人工确认 -> 选择性读取
          （直接 z.open 流式读，不落临时文件，避免解压出 13GB）

用法（实例上先 source env.sh）：
    python preprocess.py --row 167 --inspect
    python preprocess.py --row 167
    # 168 必须显式给 --seeds-per-prompt 6（默认 1）：
    #   75 prompt × 6 seed = 450 对 = blur_sd21 900 行 / blur_proc 450 行，与已发布口径一致
    python preprocess.py --row 168 --n-prompts 75 --seeds-per-prompt 6   # 需先有 SD2.1 权重，走 screen
    python preprocess.py --row 170 --list
    python preprocess.py --row 170 --suggest
    python preprocess.py --row 170 --apply --classmap indian_medicinal_170_classmap.json
    python preprocess.py --row 167 --verify
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time
import types
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq

import parquet_io
from parquet_io import ParquetWriter

WORK = Path(os.environ.get("WORK", "/root/autodl-tmp/gpu_infer"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", WORK / "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", WORK / "data"))
SCRIPT_DIR = Path(__file__).resolve().parent


def log(*a):
    print(*a, flush=True)


# ==================================================================== 167
# 数据集 style 索引 -> 模型类别索引
#   数据集 style ClassLabel: 0 Art Deco, 1 Bohemian, 2 Coastal, 3 Contemporary,
#     4 Eclectic, 5 Farmhouse, 6 Mediterranean, 7 Mid-Century Modern,
#     8 Rustic, 9 Traditional, 10 Transitional
#   模型 5 类: 0 Scandinavian / 1 bohemian / 2 coastal / 3 glamor / 4 modern
INTERIOR_KEEP = {1: 1, 2: 2, 7: 4}
MODEL_167_LABELS = {
    0: "Scandinavian furniture room style", 1: "bohemian furniture room style",
    2: "coastal furniture room style", 3: "glamor furniture room style",
    4: "modern furniture room style",
}


def _interior_raw_shards() -> list[Path]:
    raw = DATA_DIR / "interior_design"
    if not raw.is_dir():
        raise SystemExit(f"[FATAL] 原始数据集目录不存在：{raw}（先跑 fetch_assets.py --only data --items 167）")
    return sorted(raw.rglob("*.parquet"))


def row167(args) -> int:
    shards = _interior_raw_shards()
    log(f"原始分片 {len(shards)} 个：")
    for p in shards:
        log(f"  {p.name}  ({parquet_io.count_rows(p)} 行)")

    if args.inspect:
        for p in shards:
            t = pq.read_table(p)
            log(f"\n--- {p.name} ---")
            log(f"  列：{t.column_names}")
            for c in ("style", "room"):
                if c in t.column_names:
                    col = t.column(c).to_pylist()
                    dist: dict = {}
                    for v in col:
                        dist[v] = dist.get(v, 0) + 1
                    log(f"  {c} 分布：{dict(sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
        return 0

    test = [p for p in shards if p.name.startswith("test-")]
    if not test:
        log("[WARN] 没有以 test- 开头的分片，退回用全部分片（会与训练集重叠！）")
        test = shards
    else:
        log(f"使用官方 test split：{[p.name for p in test]}")

    out_dir = DATA_DIR / "interior_3cls"
    out = out_dir / "interior_3cls.parquet"

    if args.verify:
        if not out.exists():
            log(f"[BAD] 尚未产出 {out}")
            return 1
        t = pq.read_table(out)
        labs = t.column("label").to_pylist()
        dist: dict = {}
        for v in labs:
            dist[v] = dist.get(v, 0) + 1
        log(f"[OK] {out.name}: {len(labs)} 张")
        for k in sorted(dist):
            log(f"   label={k} ({MODEL_167_LABELS.get(k, '?')}) : {dist[k]} 张")
        return 0

    log(f"\n筛类 {list(INTERIOR_KEEP)} -> 模型索引 {list(INTERIOR_KEEP.values())}，写出 {out}")
    stat: dict = {}
    with ParquetWriter(out) as w:
        for p in test:
            t = pq.read_table(p)
            cols = t.column_names
            if "style" not in cols:
                raise SystemExit(f"[FATAL] {p.name} 没有 style 列，实际列={cols}")
            imgs = t.column("image").to_pylist()
            stys = t.column("style").to_pylist()
            for b, s in zip(imgs, stys):
                if s is None:
                    continue
                s = int(s)
                if s not in INTERIOR_KEEP:
                    continue
                raw = parquet_io.image_bytes(b)
                if raw is None:
                    continue
                mid = INTERIOR_KEEP[s]
                w.add(raw, mid)
                stat[mid] = stat.get(mid, 0) + 1
    log(f"完成：{out}  共 {stat and sum(stat.values())} 张  压缩后 {out.stat().st_size/1e6:.1f} MB")
    for k in sorted(stat):
        log(f"  label={k} ({MODEL_167_LABELS.get(k, '?')}) : {stat[k]} 张")
    log(f"其中模型类别 0/3（Scandinavian/glamor）无对应数据，本次评测覆盖 3/5 类。")
    return 0


# ==================================================================== 168
# 固定 prompt 列表：写进 prompts.txt 保证可复现。
# 刻意覆盖多种主体，避免单一主题让分类器靠「内容」而非「伪影」分辨。
PROMPTS_168 = [
    "a photo of a red apple on a wooden table",
    "a photo of a golden retriever running on grass",
    "a photo of a mountain landscape at sunrise",
    "a portrait of an elderly man with a beard",
    "a bowl of fresh fruit on a kitchen counter",
    "a photo of a vintage car parked on a street",
    "a close-up photo of a purple flower",
    "a photo of a modern glass office building",
    "a photo of a tabby cat sleeping on a sofa",
    "a plate of spaghetti with tomato sauce",
    "a photo of a sailboat on a calm sea",
    "a city street at night with neon signs",
    "a photo of a snowy forest in winter",
    "a cup of coffee with latte art on a table",
    "a photo of a zebra standing in savanna grass",
    "a bunch of sunflowers in a glass vase",
    "a photo of an old wooden barn in a field",
    "a close-up photo of a human eye",
    "a photo of a busy farmers market",
    "a stack of books on a wooden desk",
    "a photo of a lighthouse on a rocky coast",
    "a bowl of ramen with an egg on top",
    "a photo of a butterfly on a green leaf",
    "a photo of a desert dune under a blue sky",
    "a portrait of a young woman with curly hair",
    "a photo of a bicycle leaning against a wall",
    "a close-up photo of a honeybee on a flower",
    "a photo of a waterfall in a tropical jungle",
    "a photo of a chess board mid game",
    "a plate of sushi on a slate board",
    "a photo of a hot air balloon in the sky",
    "a photo of a rustic kitchen interior",
    "a photo of a fox in a snowy field",
    "a photo of a wooden rowboat on a lake",
    "a close-up photo of a dew-covered spiderweb",
    "a photo of a modern living room with a sofa",
    "a photo of a steam train on a bridge",
    "a basket of freshly picked strawberries",
    "a photo of a coral reef with tropical fish",
    "a photo of a street cafe in europe",
    "a close-up photo of a cat's green eye",
    "a photo of an astronaut helmet on a stand",
    "a photo of a windmill in a tulip field",
    "a photo of a potted cactus on a windowsill",
    "a photo of a fisherman on a wooden pier",
    "a photo of a colorful hot air balloon festival",
    "a photo of a snowy mountain cabin at dusk",
    "a photo of a plate of pancakes with syrup",
    "a photo of a gothic cathedral facade",
    "a photo of a squirrel eating a nut",
    "a photo of a race track with a formula car",
    "a close-up photo of a butterfly wing pattern",
    "a photo of an open laptop on a cafe table",
    "a photo of a horse galloping on a beach",
    "a photo of a neon city skyline at night",
    "a photo of a hummingbird at a feeder",
    "a photo of a stack of vinyl records",
    "a photo of a bamboo forest path",
    "a photo of a whale breaching the ocean surface",
    "a photo of a antique pocket watch",
    "a photo of a bowl of green salad",
    "a photo of a red telephone booth in london",
    "a photo of a tropical beach with palm trees",
    "a photo of a snowy owl perched on a branch",
    "a photo of a wooden pier at sunset",
    "a photo of a potter shaping a clay vase",
    "a photo of a lighthouse in a storm",
    "a photo of a field of lavender in provence",
    "a photo of a rustic bread loaf on a board",
    "a photo of a koala in a eucalyptus tree",
    "a photo of a mountain lake reflection",
    "a photo of a classic motorcycle on a road",
    "a photo of a greenhouse full of plants",
    "a photo of a bowl of colorful macarons",
    "a photo of a snowy pine forest path",
]


def _pil_to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _find_sd21() -> tuple[Path, Path]:
    d = MODEL_DIR / "sd21_base"
    if not d.is_dir():
        raise SystemExit(f"[FATAL] 没有 SD2.1 权重目录 {d}（先跑 fetch_assets.py --only aux --items 168）")
    ckpts = sorted(d.rglob("*.ckpt"))
    yamls = sorted(d.rglob("*.yaml"))
    if not ckpts:
        raise SystemExit(f"[FATAL] {d} 下没有 .ckpt")
    return ckpts[0], (yamls[0] if yamls else None)


def _install_pl_stubs() -> None:
    """让 pickle 能解析 `pytorch_lightning.*` 里的类名，而不必真装这个包。

    背景：`ckpt/stable-diffusion-2-1-base` 的官方单文件 ckpt 用 pytorch_lightning 的
    `ModelCheckpoint` 做了序列化，unpickle 时 Python 必须能 import 到那个类，否则
    `ModuleNotFoundError: No module named 'pytorch_lightning'`。我们只需要里面的
    `state_dict`，为了一个元数据对象去装整套 pytorch-lightning（及其 torch 依赖树）
    不值得，所以挂一组「按需长类」的空壳模块。

    实现：往 sys.meta_path **追加**（不是插到最前）一个 finder —— 真装了 pytorch_lightning
    的环境仍然走真包，只有在 import 失败时才落到空壳上。
    """
    import importlib.abc
    import importlib.machinery

    if getattr(_install_pl_stubs, "_done", False):
        return

    class _Stub:
        def __init__(self, *a, **kw):
            self.__dict__.update(kw)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (_Stub,), {"__module__": self.__name__})
            setattr(self, name, cls)
            return cls

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            m = _Mod(spec.name)
            m.__path__ = []          # 声明成包，好让 pickle 继续往下钻子模块
            return m

        def exec_module(self, module):
            return None

    class _Finder(importlib.abc.MetaPathFinder):
        PREFIX = "pytorch_lightning"

        def find_spec(self, fullname, path=None, target=None):
            if fullname != self.PREFIX and not fullname.startswith(self.PREFIX + "."):
                return None
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)

    sys.meta_path.append(_Finder())
    _install_pl_stubs._done = True


def _partial_denoise(pipe, prompt, gen, total_steps: int, run_steps: int):
    """跑 total_steps 步的调度，但只执行前 run_steps 步，把中间 latent 直接解码。

    这是作者所说 "unfinished/insufficient passes from an LDM" 的真实含义 ——
    模型卡把负类叫 **"noisy"**（不是「模糊」），描述是「未跑完/步数不足的生成结果」。

    ⚠️ 不能用 `num_inference_steps=run_steps` 代替：那是「用 10 步的调度完整跑完」，
    出的图只是略粗糙，模型对它**毫无判别力**（实测 900 张只得 48.22%，且
    「判成 blurry」的比例与真实类别无关）。改成截半步后，模型 24/24 全判对。
    """
    from PIL import Image

    import torch  # 本函数在模块级，看不到 row168() 里的局部 import

    snap: dict = {}

    def _cb(p, idx, t, kw):
        if idx == run_steps - 1:
            snap["lat"] = kw["latents"].detach().clone()
        return kw

    out = pipe(prompt, num_inference_steps=total_steps, generator=gen,
               height=512, width=512, output_type="latent",
               callback_on_step_end=_cb)
    lat = snap.get("lat")
    if lat is None:
        raise SystemExit(f"[FATAL] 未截到第 {run_steps} 步的 latent（总步数 {total_steps}）")
    with torch.no_grad():
        im = pipe.vae.decode(lat / pipe.vae.config.scaling_factor, return_dict=False)[0]
    im = (im / 2 + 0.5).clamp(0, 1)[0].float().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((im * 255).round().astype("uint8"))


def row168(args) -> int:
    out_sd = DATA_DIR / "blur_sd21"
    out_proc = DATA_DIR / "blur_proc"

    if args.verify:
        rc = 0
        for d in (out_sd, out_proc):
            parq = sorted(d.rglob("*.parquet")) if d.is_dir() else []
            n = sum(parquet_io.count_rows(p) for p in parq)
            dist: dict = {}
            for p in parq:
                t = pq.read_table(p)
                for v in t.column("label").to_pylist():
                    dist[int(v)] = dist.get(int(v), 0) + 1
            log(f"[{'OK ' if n else 'BAD'}][{d.name}] {n} 张  分布 {dict(sorted(dist.items()))}"
                f"   (0=blurry, 1=clear)")
            if not n:
                rc = 1
        return rc

    import torch
    from PIL import ImageFilter
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    # SD2.1 的官方单文件 ckpt 里除了 state_dict 还塞了一个 pytorch_lightning 的
    # ModelCheckpoint 对象，torch>=2.6 的 torch.load 默认 weights_only=True 会直接拒绝
    # （Unsupported global: ModelCheckpoint）。文件是本机从 HF 官方仓库拉的、sha256
    # 已逐位校验，所以这里显式放开 weights_only。只改这一次调用链，不动别处。
    if not getattr(torch.load, "_unlocked_weights_only", False):
        _torch_load = torch.load

        def _load(*a, **kw):
            kw["weights_only"] = False
            return _torch_load(*a, **kw)

        _load._unlocked_weights_only = True
        torch.load = _load
        log("[info] 已放开 torch.load 的 weights_only（SD2.1 ckpt 含 ModelCheckpoint 对象）")

    _install_pl_stubs()
    log("[info] 已挂 pytorch_lightning 空壳模块（只为让 pickle 能解析 ModelCheckpoint）")

    ckpt, yaml = _find_sd21()
    log(f"SD2.1 权重：{ckpt}")
    log(f"官方配置  ：{yaml}")
    # `config=` 必须指向【本地】的 diffusers 格式配置目录。
    # 不传的话 from_single_file 会去 hub 上找 `stabilityai/stable-diffusion-2-1` 取
    # tokenizer / 各组件 config，而那个仓库在 hf-mirror 上是 401（gated）。
    # 我们只要配置文件（几 MB），权重仍从本地单文件 ckpt 取。
    # 该目录来自公开镜像 SfinOe/stable-diffusion-v2-1（它是 768 变体），
    # 所以 unet/config.json 的 sample_size 已按 SD2.1-base(512) 手工改成 64。
    cfg_dir = ckpt.parent / "diffusers_cfg"
    if not cfg_dir.is_dir():
        raise SystemExit(f"[FATAL] 缺 diffusers 格式配置目录 {cfg_dir}")
    log(f"本地组件配置：{cfg_dir}")
    log("加载 StableDiffusionPipeline（from_single_file + 官方 LDM 配置）...")
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_single_file(
        str(ckpt),
        config=str(cfg_dir),
        original_config_file=str(yaml) if yaml else None,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # ⚠️ 参数化必须显式设成 eps，不能沿用 diffusers_cfg 里的 v_prediction。
    # 实证：该 ckpt（SD2.1-base, 512）用 eps 能出正确的「红苹果」图；
    # 用 v_prediction 会出平涂棕色块（900 张全部报废，模型因此全判 clear 得 50%）。
    # 依据：官方 v2-inference.yaml 里【没有】parameterization 字段 → LDM 默认 eps；
    # 而 diffusers_cfg 抄自 768 版镜像 SfinOe/stable-diffusion-v2-1，那边才是 v_prediction。
    sc = dict(pipe.scheduler.config)
    sc.update(prediction_type="epsilon", beta_schedule="scaled_linear",
              steps_offset=1, clip_sample=False, set_alpha_to_one=False)
    pipe.scheduler = DDIMScheduler.from_config(sc)
    for k, v in (("prediction_type", "epsilon"), ("steps_offset", 1),
                 ("clip_sample", False), ("set_alpha_to_one", False)):
        if dict(pipe.scheduler.config).get(k) != v:
            raise SystemExit(f"[FATAL] SD2.1 scheduler 配置不符：{k}="
                             f"{dict(pipe.scheduler.config).get(k)!r}，应为 {v!r}")
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    log(f"已加载，用时 {time.time()-t0:.0f}s；scheduler={type(pipe.scheduler).__name__} "
        f"prediction_type={dict(pipe.scheduler.config).get('prediction_type')} "
        f"steps_offset={dict(pipe.scheduler.config).get('steps_offset')}")

    prompts = PROMPTS_168[: args.n_prompts]
    (out_sd.parent / "prompts_168.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    n_pairs = len(prompts) * args.seeds_per_prompt
    log(f"\n生成 {n_pairs} 组配对（{len(prompts)} 条 prompt × {args.seeds_per_prompt} 个 seed；"
        f"每组 {args.clean_steps} 步跑完出 clean，同 {args.clean_steps} 步调度只跑 "
        f"{args.blur_steps} 步解码出「半成品」blurry，同 prompt 同 seed），"
        f"外加同批 clean 图的高斯模糊对照（radius={args.blur_radius}）")
    log(f"prompt 列表已落盘：{out_sd.parent / 'prompts_168.txt'}")

    n_clean = n_blur = n_proc = 0
    t0 = time.time()
    with ParquetWriter(out_sd / "blur_sd21.parquet") as w_sd, \
         ParquetWriter(out_proc / "blur_proc.parquet") as w_pr:
        done = 0
        for i, pr in enumerate(prompts):
            for k in range(args.seeds_per_prompt):
                seed = args.seed_base + i * args.seeds_per_prompt + k
                # 同一个 seed → 同一个初始噪声；只改「跑多少步」→ 天然配对（同作者的 30/10 步设计）
                g_clean = torch.Generator(device="cpu").manual_seed(seed)
                clean = pipe(pr, num_inference_steps=args.clean_steps, generator=g_clean,
                             height=512, width=512, output_type="pil").images[0]
                # 「blurry」= 在 clean_steps 步的调度里【只跑 blur_steps 步】就解码，
                # 复现作者所说的 unfinished / insufficient passes（见 _partial_denoise 注释）
                g_blur = torch.Generator(device="cpu").manual_seed(seed)
                blurry = _partial_denoise(pipe, pr, g_blur,
                                          args.clean_steps, args.blur_steps)

                # 模型 config：0=blurry, 1=clear
                w_sd.add(_pil_to_png_bytes(clean), 1)
                w_sd.add(_pil_to_png_bytes(blurry), 0)
                n_clean += 1
                n_blur += 1

                # 对照：clean 图做光学高斯模糊 —— 语义上「是模糊的」，label=0
                # 若模型真在检测 SD 未收敛伪影而非通用模糊，这里会被大量判成 clear
                proc = clean.convert("RGB").filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
                w_pr.add(_pil_to_png_bytes(proc), 0)
                n_proc += 1

                done += 1
                if done % 20 == 0 or done == n_pairs:
                    el = time.time() - t0
                    log(f"  [{done}/{n_pairs}] clean={n_clean} blurry={n_blur} proc_blur={n_proc} "
                        f"用时 {el:.0f}s（{el/done:.2f}s/组）")

    log(f"\n完成：")
    log(f"  SD2.1 集 {out_sd/'blur_sd21.parquet'}  clean={n_clean} blurry={n_blur}（label 0=blurry,1=clear）")
    log(f"  对照组   {out_proc/'blur_proc.parquet'}  {n_proc} 张（全部是 clean 图的高斯模糊版，label=0）")
    return 0


# ==================================================================== 170
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _med_zip() -> Path:
    d = DATA_DIR / "_med_zip"
    zs = sorted(d.rglob("*.zip")) if d.is_dir() else []
    if not zs:
        raise SystemExit(f"[FATAL] 找不到 med.zip（先跑 fetch_assets.py --only data --items 170）查过目录 {d}")
    return zs[0]


def _model170_labels() -> dict[int, str]:
    cfg = MODEL_DIR / "dima806__medicinal_plants_image_detection" / "config.json"
    if not cfg.exists():
        raise SystemExit(f"[FATAL] 找不到 {cfg}")
    d = json.loads(cfg.read_text(encoding="utf-8"))
    i2l = d.get("id2label") or {}
    return {int(k): v for k, v in i2l.items()}


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _zip_class_dirs(z: zipfile.ZipFile) -> dict[str, list[str]]:
    """按【图片所在目录的完整相对路径】归类 zip 里的图片条目。

    注意不是「第一层目录」——med.zip 的布局是 `all_dataset/<类名>/xxx.jpg`，
    按第一层归类会把 20798 张图全塞进同一个 `all_dataset` 桶里。
    这里用「文件所在目录」做 key，与映射表的 `dir` 字段
    （形如 `all_dataset/Curry Leaf-Murraya koenigii`）一一对应。
    """
    out: dict[str, list[str]] = {}
    for name in z.namelist():
        if name.endswith("/") or not name.lower().endswith(IMG_EXT):
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        out.setdefault("/".join(parts[:-1]), []).append(name)
    return out


def _dir_leaf(d: str) -> str:
    """`all_dataset/Curry Leaf-Murraya koenigii` -> `Curry Leaf-Murraya koenigii`。"""
    return d.rsplit("/", 1)[-1]


def row170(args) -> int:
    zp = _med_zip()
    log(f"med.zip: {zp}  ({zp.stat().st_size/1e9:.2f} GB)")

    with zipfile.ZipFile(zp) as z:
        dirs = _zip_class_dirs(z)
    leaf2dir = {_dir_leaf(d): d for d in dirs}
    log(f"图片目录 {len(dirs)} 个（按「图片所在目录」归类）")

    labels = _model170_labels()
    log(f"模型类别 {len(labels)} 个")

    if args.list:
        for k in sorted(dirs):
            log(f"  {k:<72} {len(dirs[k]):>6} 张")
        return 0

    # ---------------------------------------------------------------- suggest
    # 只做字符串归一化 + 相似度，不理解语义。170 的实际映射表是人工逐条核过的
    # （indian_170_classmap.json），这条路径留作可复核的旁证/复现用。
    if args.suggest:
        rows = []
        used: dict[int, str] = {}
        for k in sorted(dirs):
            nk = _norm(_dir_leaf(k))
            cands = []
            for idx, lab in sorted(labels.items()):
                nl = _norm(lab)
                if not nl:
                    continue
                ratio = SequenceMatcher(None, nk, nl).ratio()
                if nk == nl:
                    ratio = 1.0
                elif nk in nl or nl in nk:
                    ratio = max(ratio, 0.92)
                cands.append({"model_idx": idx, "model_label": lab, "score": round(ratio, 3)})
            cands.sort(key=lambda c: -c["score"])
            best = cands[0] if cands else None
            rows.append({
                "dataset_dir": k,
                "n_images": len(dirs[k]),
                "suggested_model_idx": best["model_idx"] if best else None,
                "suggested_model_label": best["model_label"] if best else None,
                "score": best["score"] if best else 0.0,
                "top3": cands[:3],
                "needs_review": (best is None) or (best["score"] < args.threshold),
            })
            if best is not None and best["score"] >= args.threshold:
                used.setdefault(best["model_idx"], k)

        unmapped = [i for i in sorted(labels) if i not in used]
        payload = {
            "_comment": [
                "170 的 zip 目录 -> 模型类别 候选映射（半自动，待人工确认）。",
                "suggest 只做字符串归一化 + 相似度，不理解语义，错拼/同名类必须人工过一遍：",
                "  已知易错：Curry(1) vs Curry_Leaf(10)、Guava(36) vs Gauva(31)、",
                "            Tulsi(18) vs Tulasi(46)、Bhrami(37) vs Brahmi(51)、",
                "            Doddpathre(15) vs Doddapatre(27)、Papaya(50) vs Pappaya(9)",
                "这 6 对其实是【同一个目录】：模型把同一株植物拆成了两个输出类，",
                "正式映射表因此用 mode=groups 把它们并成同一簇（indian_170_classmap.json）。",
            ],
            "zip": str(zp),
            "threshold": args.threshold,
            "model_labels": {str(k): v for k, v in sorted(labels.items())},
            "mapping": [
                {**r, "expected_model_idx": r["suggested_model_idx"] if not r["needs_review"] else None}
                for r in rows
            ],
            "model_classes_without_dataset_dir": {str(i): labels[i] for i in unmapped},
        }
        out = Path(args.classmap) if os.path.isabs(args.classmap) else SCRIPT_DIR / args.classmap
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok = sum(1 for r in rows if not r["needs_review"])
        log(f"\n候选映射已写 {out}")
        log(f"  高置信（score>={args.threshold}）{n_ok}/{len(rows)} 个目录")
        log(f"  待人工确认 {len(rows)-n_ok} 个；模型侧无对应目录的类 {len(unmapped)} 个")
        return 0

    # ---------------------------------------------------------------- apply
    if args.apply:
        cm_path = Path(args.classmap) if os.path.isabs(args.classmap) else SCRIPT_DIR / args.classmap
        if not cm_path.exists():
            raise SystemExit(f"[FATAL] 找不到映射表 {cm_path}")
        cm = json.loads(cm_path.read_text(encoding="utf-8"))
        mode = (cm.get("mode") or "perm").strip().lower()

        # 统一成 [(簇号, 簇名, [模型类别索引], 盘内目录)] —— 两种映射表格式都收
        sel: list[tuple[int, str, list[int], str]] = []
        if mode == "groups":
            for g in sorted(cm.get("groups") or [], key=lambda x: int(x["group"])):
                gi = int(g["group"])
                mids = [int(m) for m in g["model_idx"]]
                d = g["dir"] if "/" in g["dir"] else leaf2dir.get(g["dir"], g["dir"])
                sel.append((gi, g.get("label") or "/".join(labels.get(m, "?") for m in mids),
                            mids, d))
            ids = [s[0] for s in sel]
            if ids != list(range(len(sel))):
                raise SystemExit(f"[FATAL] {cm_path.name}: 簇号必须是连续的 0..{len(sel)-1}，实际开头 {ids[:8]}")
        else:
            # 旧的 v1 候选表：一条 dir -> 一个模型类，等价于每簇只有一个类
            for r in cm.get("mapping") or []:
                idx = r.get("expected_model_idx")
                if idx is None:
                    continue
                d = r["dataset_dir"]
                sel.append((len(sel), labels.get(int(idx), str(idx)), [int(idx)], d))
            if not sel:
                raise SystemExit(f"[FATAL] {cm_path.name} 里没有任何可用映射")

        # 目录存在性检查（映射表是人写的，必须逐条验证）
        bad = [s[3] for s in sel if s[3] not in dirs]
        if bad:
            for b in bad[:10]:
                hint = [k for k in leaf2dir if _norm(k) == _norm(_dir_leaf(b))]
                log(f"  [!] 目录不存在：{b}" + (f"  （同名候选：{hint[:3]}）" if hint else ""))
            raise SystemExit(f"[FATAL] {len(bad)}/{len(sel)} 个目录在 zip 里找不到，先修映射表")
        covered = {m for _, _, mids, _ in sel for m in mids}
        log(f"映射表 {cm_path.name} [{mode}]：{len(sel)} 簇，覆盖模型类 {len(covered)}/{len(labels)}")
        miss_cls = sorted(set(labels) - covered)
        if miss_cls:
            log(f"  [!] 未覆盖的模型类 {len(miss_cls)} 个：{[(i, labels[i]) for i in miss_cls]}")

        # 目录名用行号而不是类别数：模型 52 类、评测集 46 簇，叫 indian_52cls 会误导
        out_dir = DATA_DIR / "indian_170"
        out = out_dir / "indian_170.parquet"
        log(f"\n直接从 zip 流式读取（不解压），每簇最多 {args.per_class} 张 → {out}")
        log(f"  parquet 的 label 写的是【簇号 0..{len(sel)-1}】（groups 模式要求，不是模型索引）")

        rng = random.Random(args.sample_seed)
        stat: dict[int, int] = {}
        skipped = 0
        t0 = time.time()
        with zipfile.ZipFile(zp) as z, ParquetWriter(out) as w:
            for gi, gname, mids, d in sorted(sel, key=lambda s: -len(dirs.get(s[3], []))):
                members = sorted(dirs[d])
                if len(members) > args.per_class:
                    members = sorted(rng.sample(members, args.per_class))
                got = 0
                for m in members:
                    try:
                        with z.open(m) as fh:
                            raw = fh.read()
                    except Exception:  # noqa: BLE001
                        skipped += 1
                        continue
                    if not raw:
                        skipped += 1
                        continue
                    w.add(raw, gi)
                    got += 1
                stat[gi] = got
                tag = f"model[{','.join(str(m) for m in mids)}]"
                log(f"  {_dir_leaf(d):<44} -> 簇{gi:>2} {tag:<12} {gname[:34]:<36} {got:>4} 张")
        n_img = sum(stat.values())
        log(f"\n完成：{out}  {n_img} 张 / 覆盖 {len(stat)}/{len(sel)} 个簇  "
            f"用时 {time.time()-t0:.0f}s  跳过 {skipped} 张")
        empty = [g for g, c in stat.items() if c == 0]
        if empty:
            log(f"[WARN] 空簇 {empty}")
        # 顺带把「每簇张数」回写到映射表旁，便于报告引用
        side = out.with_name("indian_170_counts.json")
        side.write_text(json.dumps(
            {"n_images": n_img, "per_class_cap": args.per_class, "sample_seed": args.sample_seed,
             "counts": {str(g): stat.get(g, 0) for g, _, _, _ in sel}}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        log(f"每簇张数已写 {side}")
        log(f"压缩后大小 {out.stat().st_size/1e6:.1f} MB")
        return 0

    log("[WARN] 170 需要指定动作：--list / --suggest / --apply")
    return 2


# ==================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="167/168/170 评测集预处理（产出 parquet，供 run_inference.py 直接吃）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--row", type=int, required=True, choices=[167, 168, 170])
    ap.add_argument("--inspect", action="store_true", help="167：打印原始数据集 schema/分布")
    ap.add_argument("--list", action="store_true", help="170：列出 zip 里的类目录与张数")
    ap.add_argument("--suggest", action="store_true", help="170：生成候选映射表")
    ap.add_argument("--apply", action="store_true", help="170：按确认后的映射表建评测集")
    ap.add_argument("--verify", action="store_true", help="167/168：校验已产出的评测集")
    ap.add_argument("--classmap", default="indian_170_classmap.json")
    ap.add_argument("--threshold", type=float, default=0.80, help="170 suggest 的高置信阈值")
    ap.add_argument("--per-class", type=int, default=80, help="170 每簇最多取多少张")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="170 每簇抽样的随机种子（固定后评测集可复现）")
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS_168), help="168 生成多少组")
    ap.add_argument("--seeds-per-prompt", type=int, default=1,
                    help="168 每条 prompt 用几个 seed。本表口径 = 6（75×6 = 450 对 = 900 行），"
                         "deploy_all.sh 已自动带上；手敲务必加 --seeds-per-prompt 6，"
                         "否则只生成 75 对（150 行）= 1/6 数据")
    ap.add_argument("--clean-steps", type=int, default=30)
    ap.add_argument("--blur-steps", type=int, default=10,
                    help="168「blurry」在 clean-steps 步的调度里只跑多少步就解码（半成品 latent）")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--blur-radius", type=float, default=3.0, help="168 对照用高斯模糊半径")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    log(f"WORK={WORK}  MODEL_DIR={MODEL_DIR}  DATA_DIR={DATA_DIR}")
    fn = {167: row167, 168: row168, 170: row170}[args.row]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
