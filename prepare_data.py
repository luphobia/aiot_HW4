import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _unique_path(dst_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / (base + ext)
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate


def _find_class_folders(root: Path) -> List[Path]:
    """
    找出像 ImageFolder 的 class folder：資料夾底下「直接」有圖片檔者。
    """
    candidates = []
    for d in root.rglob("*"):
        if d.is_dir():
            try:
                if any(_is_image(x) for x in d.iterdir() if x.is_file()):
                    candidates.append(d)
            except Exception:
                continue
    return candidates


def _choose_best_parent(class_folders: List[Path]) -> List[Path]:
    """
    很多 Kaggle 資料集會有 train/valid/test 之類的結構，
    我們挑「同一個 parent 底下 classes 數量最多」那一組當來源。
    """
    by_parent: Dict[Path, List[Path]] = {}
    for d in class_folders:
        by_parent.setdefault(d.parent, []).append(d)

    best_parent = max(by_parent.items(), key=lambda kv: len(kv[1]))[0]
    return by_parent[best_parent]


def _collect_images_from_class_folders(class_folders: List[Path]) -> Dict[str, List[Path]]:
    """
    回傳：{label: [img_paths...]}
    label = class folder 名稱
    """
    mapping: Dict[str, List[Path]] = {}
    for cls_dir in class_folders:
        label = cls_dir.name
        imgs = [p for p in cls_dir.iterdir() if p.is_file() and _is_image(p)]
        if imgs:
            mapping.setdefault(label, []).extend(imgs)
    return mapping


def download_and_prepare(
    dataset_slug: str = "anshtanwar/pets-facial-expression-dataset",
    out_dir: str = "data",
    val_ratio: float = 0.15,
    seed: int = 42,
    reset: bool = True,
):
    """
    下載 Kaggle dataset（kagglehub）→ 自動找 class folders → 複製成 ImageFolder:
      out_dir/train/<label>/*.jpg
      out_dir/val/<label>/*.jpg

    - reset=True：會清掉 out_dir 重新建立（你要「重新下載就好」建議用 True）
    """
    out_dir = Path(out_dir)
    out_train = out_dir / "train"
    out_val = out_dir / "val"

    if reset:
        _reset_dir(out_train)
        _reset_dir(out_val)
    else:
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)

    # 下載（kagglehub 會 cache 到本機；你重跑也會很快）
    import kagglehub
    dl_path = Path(kagglehub.dataset_download(dataset_slug))

    # 找 class folders
    candidates = _find_class_folders(dl_path)
    if not candidates:
        raise RuntimeError(f"在下載內容中找不到類別資料夾（資料夾底下直接有圖片）。下載路徑：{dl_path}")

    class_folders = _choose_best_parent(candidates)

    label_to_imgs = _collect_images_from_class_folders(class_folders)
    if not label_to_imgs:
        raise RuntimeError("找到類別資料夾但沒有任何圖片（或格式不支援）。")

    rng = random.Random(seed)
    stats = {"total": 0, "train": 0, "val": 0}

    # 依 label 切分 train/val，並複製到 out_dir
    for label, imgs in label_to_imgs.items():
        imgs = [p for p in imgs if p.is_file() and _is_image(p)]
        rng.shuffle(imgs)

        n_val = int(len(imgs) * val_ratio)
        val_set = set(imgs[:n_val])
        train_list = [p for p in imgs if p not in val_set]
        val_list = list(val_set)

        # 建類別資料夾
        (out_train / label).mkdir(parents=True, exist_ok=True)
        (out_val / label).mkdir(parents=True, exist_ok=True)

        for p in train_list:
            dst = _unique_path(out_train / label, p.name)
            shutil.copy2(str(p), str(dst))
            stats["total"] += 1
            stats["train"] += 1

        for p in val_list:
            dst = _unique_path(out_val / label, p.name)
            shutil.copy2(str(p), str(dst))
            stats["total"] += 1
            stats["val"] += 1

    # 寫 meta，方便你檢查 classes
    meta = {
        "dataset_slug": dataset_slug,
        "out_dir": str(out_dir.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "labels": sorted(label_to_imgs.keys()),
        "stats": stats,
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Prepared dataset ===")
    print("Train:", out_train.resolve())
    print("Val:", out_val.resolve())
    print("Meta:", (out_dir / "prepare_meta.json").resolve())
    print("Labels:", meta["labels"])
    print("Stats:", stats)

    return str(out_dir.resolve())


if __name__ == "__main__":
    # 直接跑這支也可：會重新建立 data/
    download_and_prepare(
        dataset_slug="anshtanwar/pets-facial-expression-dataset",
        out_dir="data",
        val_ratio=0.15,
        seed=42,
        reset=True,
    )
