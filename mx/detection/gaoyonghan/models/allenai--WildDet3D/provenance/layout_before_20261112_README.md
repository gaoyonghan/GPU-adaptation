# allenai/WildDet3D

Run `python inference/run.py` from this model folder. The native script and its support files are in `inference/native/`; format conversion and scoring entry points are in `evaluation/`. Weights are in `weights/`, and the dataset belongs under the project-level `dataset/wilddet3d_val/`. Relative model paths are prepared by `inference/model_paths.py`.

An external symbolic link in the original upstream source tree pointed to a dataset outside the project. That link was omitted from this independent model folder. The dataset files are maintained under the project's `dataset/` directory instead. The server dataset transfer is tracked separately in the project's `上传状态.json`. No full run of this reorganized folder has been performed.
