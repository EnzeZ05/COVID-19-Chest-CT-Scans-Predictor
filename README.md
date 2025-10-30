# COVID-19 Chest CT Scans Predictor

Supervised learning pipeline for classifying chest CT images into clinically relevant categories (e.g., COVID-19 vs. non-COVID). Includes concise data utilities, a readable PyTorch training loop, evaluation, and inference.

---

## Features

- Minimal, reproducible PyTorch training loop
- Baseline CNN plus torchvision backbones (e.g., ResNet-18, MobileNetV3)
- Deterministic switches (fixed seeds, options for CuDNN determinism)
- Train/val split and optional CSV-based dataset
- Metrics: accuracy, precision/recall/F1, AUROC; confusion matrix + ROC/PR curves
- Checkpointing, early stopping, and simple inference CLI

---

## Project Structure

```
.
├─ data/                        # Place datasets here (gitignored)
│  ├─ raw/                      # Original images or slices
│  ├─ processed/                # Output of split (train/val[/test])
│  └─ meta.csv                  # Optional CSV with filepath,label
├─ models/                      # Weights, logs (gitignored)
├─ src/
│  ├─ datasets.py               # Dataset + transforms
│  ├─ model_zoo.py              # cnn_small + torchvision backbones
│  ├─ train.py                  # Train loop
│  ├─ evaluate.py               # Eval + plots
│  ├─ infer.py                  # Inference entrypoint
│  └─ utils.py                  # Seed, I/O, metrics helpers
├─ configs/
│  └─ default.yaml              # Hyperparameters
├─ requirements.txt
└─ README.md
```

---

## Setup

### 1) Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Minimal `requirements.txt`:

```
torch>=2.2
torchvision>=0.17
numpy
pandas
scikit-learn
opencv-python
matplotlib
tqdm
pyyaml
```

### 2) Data Layout

Choose one:

**A. Folders per class**
```
data/raw/
├─ covid/
│  ├─ img_001.png
│  └─ ...
└─ non_covid/
   ├─ img_101.png
   └─ ...
```

**B. CSV**
```
data/meta.csv   # columns: filepath,label
```

If you start from volumes (DICOM/NIfTI), convert to 2D slices or MIPs consistently (windowing, spacing) before training.

---

## Quickstart

### Split

```bash
python -m src.utils split \
  --input_dir data/raw \
  --output_dir data/processed \
  --val_ratio 0.2 \
  --seed 42
```

(Adapt `src/utils.py` if using `data/meta.csv` so it writes `train.csv` and `val.csv`.)

### Train

```bash
python -m src.train \
  --config configs/default.yaml \
  --data_dir data/processed \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --model cnn_small \
  --img_size 224 \
  --num_workers 4 \
  --seed 42
```

Switch to a backbone:

```bash
python -m src.train --config configs/default.yaml --model resnet18 --lr 1e-3
```

### Evaluate

```bash
python -m src.evaluate \
  --data_dir data/processed \
  --weights models/best.pt \
  --split val \
  --img_size 224
```

### Inference

```bash
# Single image
python -m src.infer --weights models/best.pt --image path/to/ct.png

# Folder of images
python -m src.infer --weights models/best.pt --folder path/to/dir
```

---

## Configuration (example)

`configs/default.yaml`:

```yaml
data:
  train_dir: data/processed/train
  val_dir:   data/processed/val
  img_size:  224
  num_workers: 4

train:
  epochs: 20
  batch_size: 32
  lr: 3e-4
  weight_decay: 1e-4
  optimizer: adamw
  scheduler: cosine
  early_stop: 7

model:
  name: cnn_small            # cnn_small, resnet18, mobilenet_v3_small
  num_classes: 2
  dropout: 0.2

seed: 42
mixed_precision: true
```

---

## Preprocessing Notes

- Resize all images to a fixed size (default 224×224). Center-crop if needed.
- If using pretrained backbones, normalize with ImageNet stats:
  - mean: [0.485, 0.456, 0.406]
  - std:  [0.229, 0.224, 0.225]
- Recommended augmentations: light flips/rotations and mild brightness/contrast jitter. Avoid heavy distortions that may hide pathology.

---

## Baseline Model

A compact CNN to validate the pipeline before deeper models:

- Conv-ReLU-Pool × 2 → Flatten → Linear-ReLU → Linear
- Cross-entropy loss
- Optional BatchNorm and Dropout in `src/model_zoo.py`

If loss diverges: lower `--lr` (e.g., `1e-4`), enable BatchNorm, verify label mapping and normalization.

---

## Reproducibility

- Set seeds (`torch`, `numpy`, `random`) and optionally enforce CuDNN determinism.
- Save `configs/default.yaml` and the git commit hash for each run.
- Freeze environment:
  ```bash
  pip freeze > models/last_run/requirements.lock.txt
  ```

---

## Results (template)

| Model            | Img Size | Params | Val Acc | AUROC | F1   |
|------------------|----------|--------|---------|-------|------|
| cnn_small        | 224      | 1.1M   | 0.XX    | 0.XX  | 0.XX |
| resnet18         | 224      | 11.7M  | 0.XX    | 0.XX  | 0.XX |
| mobilenet_v3_s   | 224      | 2.5M   | 0.XX    | 0.XX  | 0.XX |

Populate from `src.evaluate`.

---

## .gitignore Suggestion

```
# data and models
data/
models/
*.pt
*.ckpt

# caches and logs
*.log
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.lightning_logs/

# environments
.venv/
.env/
```

---

## License

MIT License

---
