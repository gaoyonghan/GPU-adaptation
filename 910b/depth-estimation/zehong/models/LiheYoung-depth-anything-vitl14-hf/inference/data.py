from pathlib import Path
def _discover_samples(name, root, sample_list=None):
    # A pair manifest lets the caller select an exact split instead of scanning.
    # Each line: RGB_PATH DEPTH_PATH [optional metadata]; paths relative to root.
    if sample_list is not None:
        list_path = Path(sample_list)
        if not list_path.is_absolute():
            list_path = root / list_path
        samples = []
        for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{list_path}:{line_number}: expected RGB_PATH DEPTH_PATH")
            image_path, depth_path = (root / value.lstrip("/\\") for value in fields[:2])
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Missing pair: {image_path}, {depth_path}")
            sample = {"id": f"{line_number:06d}_{image_path.stem}", "image": image_path, "depth": depth_path}
            if name == "DIODE":
                sample["mask"] = depth_path.with_name(depth_path.name.replace("_depth.npy", "_depth_mask.npy"))
            samples.append(sample)
        return samples
    if name == "KITTI":
        samples = []
        for depth_path in sorted(root.glob("**/proj_depth/groundtruth/image_02/*.png")):
            drive = depth_path.parents[3].name
            date = drive.split("_drive_", 1)[0]
            image_path = root / date / drive / "image_02" / "data" / depth_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing KITTI RGB image: {image_path}")
            samples.append(
                {
                    "id": f"{drive}/{depth_path.stem}",
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "NYUv2":
        list_path = root / "filename_list_test.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing NYUv2 test list: {list_path}")
        samples = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Expected RGB/depth pair in {list_path}: {line}")
            image_rel, depth_rel = fields[:2]
            image_path = root / image_rel.lstrip("/\\")
            depth_path = root / depth_rel.lstrip("/\\")
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Missing NYUv2 pair: {image_path}, {depth_path}")
            samples.append(
                {
                    "id": f"{len(samples):06d}_{image_path.stem}",
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    if name == "DIODE":
        samples = []
        for image_path in sorted((root / "val").glob("**/*.png")):
            depth_path = image_path.with_name(f"{image_path.stem}_depth.npy")
            mask_path = image_path.with_name(f"{image_path.stem}_depth_mask.npy")
            if not depth_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing DIODE depth or mask: {depth_path}, {mask_path}"
                )
            samples.append(
                {
                    "id": str(image_path.relative_to(root).with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                    "mask": mask_path,
                }
            )
        return samples

    if name == "Sintel":
        samples = []
        image_root = root / "training" / "final"
        depth_root = root / "training" / "depth"
        for image_path in sorted(image_root.glob("**/*.png")):
            relative_path = image_path.relative_to(image_root)
            depth_path = (depth_root / relative_path).with_suffix(".dpt")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing Sintel depth: {depth_path}")
            samples.append(
                {
                    "id": str(relative_path.with_suffix("")),
                    "image": image_path,
                    "depth": depth_path,
                }
            )
        return samples

    raise ValueError(f"Unsupported dataset: {name}")
