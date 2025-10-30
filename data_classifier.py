import os
from pathlib import Path
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

data_root = Path("COVID-19_Radiography_Dataset")
out_root  = Path("Covid_Data")
class_dirs = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

train_ratio, val_ratio, test_ratio = 0.75, 0.15, 0.10
seed = 42

img_exts = {".png", ".jpg", ".jpeg"}

def collect_images(root):
    paths, labels = [], []
    for cls in class_dirs:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {cls_dir}")
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in img_exts]
        paths.extend(imgs)
        labels.extend([cls] * len(imgs))
    return paths, labels

def link_or_copy(src, dst):
    dst.parent.mkdir(parents = True, exist_ok = True)
    try:
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    except Exception:
        shutil.copy2(src, dst)

def write_split(root_out, subset_name, idx_list, paths, labels):
    for i in idx_list:
        src = paths[i]
        cls = labels[i]
        dst = root_out / subset_name / cls / src.name
        link_or_copy(src, dst)

def print_counts(title, idxs, labels):
    sub = [labels[i] for i in idxs]
    c = Counter(sub)
    total = len(sub)
    parts = " | ".join([f"{k}:{c[k]}" for k in sorted(c)])
    print(f"{title:>5}: {total:5d} | {parts}")

def main():
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train+val+test must sum to 1.0")

    paths, labels = collect_images(data_root)
    if not paths:
        raise SystemExit("no images found — check data_root/class_dirs")

    sss1 = StratifiedShuffleSplit(n_splits = 1, test_size = (1.0 - train_ratio), random_state = seed)
    train_idx, temp_idx = next(sss1.split(paths, labels))

    remain = 1.0 - train_ratio
    val_within = val_ratio / remain
    sss2 = StratifiedShuffleSplit(n_splits = 1, test_size = (1.0 - val_within), random_state = seed)

    val_rel, test_rel = next(
        sss2.split([paths[i] for i in temp_idx], [labels[i] for i in temp_idx])
    )
    val_idx  = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]

    for sub in ("train", "val", "test"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    write_split(out_root, "train", train_idx, paths, labels)
    write_split(out_root, "val", val_idx, paths, labels)
    write_split(out_root, "test", test_idx, paths, labels)

    print(f"done -> {out_root}")
    print_counts("all", range(len(paths)), labels)
    print_counts("train", train_idx, labels)
    print_counts("val",   val_idx, labels)
    print_counts("test",  test_idx, labels)

if __name__ == "__main__":
    main()
