---
dataset_info:
  features:
  - name: image_id
    dtype: int64
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: image
    dtype: image
  - name: objects
    sequence:
    - name: bbox_id
      dtype: int64
    - name: category
      dtype:
        class_label:
          names:
            '0': bag
            '1': bottom
            '2': dress
            '3': hat
            '4': outer
            '5': shoes
            '6': top
    - name: bbox
      sequence: float64
      length: 4
    - name: area
      dtype: int64
  splits:
  - name: train
    num_bytes: 4955239128.125
    num_examples: 84431
  - name: test
    num_bytes: 261907687.0
    num_examples: 5544
  download_size: 5191163522
  dataset_size: 5217146815.125
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
