import argparse
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


@dataclass
class DetDecision:
    species: str            # "dog" / "cat" / "unknown" / "ambiguous"
    best_conf: float
    dog_conf: float
    cat_conf: float
    bbox: Optional[Tuple[int, int, int, int]]


class SpeciesSplitter:
    """
    使用 YOLOv8 COCO 預訓練模型偵測 dog/cat，並做資料分流。
    COCO id: cat=15, dog=16
    """
    def __init__(self, yolo_model: str = "yolov8n.pt"):
        self.model = YOLO(yolo_model)

    def decide_species(
        self,
        img: Image.Image,
        conf_thres: float = 0.25,
        ambiguous_margin: float = 0.05,
        prefer: Optional[str] = None,   # "dog" or "cat" or None
    ) -> DetDecision:
        w, h = img.size
        im_np = np.array(img)

        r = self.model.predict(im_np, conf=conf_thres, verbose=False)[0]
        dog_best = 0.0
        cat_best = 0.0
        best_bbox = None

        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                if cls_id not in (15, 16):
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if cls_id == 16:  # dog
                    if conf > dog_best:
                        dog_best = conf
                        best_bbox = (x1, y1, x2, y2)
                else:  # cat
                    if conf > cat_best:
                        cat_best = conf
                        best_bbox = (x1, y1, x2, y2)

        if dog_best == 0.0 and cat_best == 0.0:
            return DetDecision("unknown", 0.0, 0.0, 0.0, None)

        # 若兩者都出現且很接近 → ambiguous（避免誤分）
        if dog_best > 0 and cat_best > 0 and abs(dog_best - cat_best) <= ambiguous_margin:
            # 若有 prefer，就把 ambiguous 交給 prefer（可選）
            if prefer in ("dog", "cat"):
                chosen = prefer
            else:
                return DetDecision("ambiguous", max(dog_best, cat_best), dog_best, cat_best, best_bbox)
        else:
            chosen = "dog" if dog_best > cat_best else "cat"

        return DetDecision(chosen, max(dog_best, cat_best), dog_best, cat_best, best_bbox)


def iter_images_by_split(src_root: Path):
    """
    預期 src_root 結構：
      src_root/train/<label>/*.jpg
      src_root/val/<label>/*.jpg
    """
    for split in ("train", "val"):
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_path in label_dir.iterdir():
                if img_path.is_file() and is_image(img_path):
                    yield split, label, img_path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def unique_name(dst_dir: Path, filename: str) -> Path:
    """
    避免檔名衝突：若已存在就加 _1, _2...
    """
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / (base + ext)
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data", help="來源資料夾（含 train/val）")
    parser.add_argument("--out_dog", type=str, default="data_dog")
    parser.add_argument("--out_cat", type=str, default="data_cat")
    parser.add_argument("--out_unknown", type=str, default="data_unknown")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 偵測信心閾值")
    parser.add_argument("--amb_margin", type=float, default=0.05, help="dog/cat 置信度差距小於此值視為 ambiguous")
    parser.add_argument("--prefer", type=str, default="none", choices=["none", "dog", "cat"], help="遇到 ambiguous 時偏好哪個物種")
    parser.add_argument("--yolo", type=str, default="yolov8n.pt")
    parser.add_argument("--copy", action="store_true", help="複製檔案（預設）")
    parser.add_argument("--move", action="store_true", help="搬移檔案（會從來源移除）")
    parser.add_argument("--dry_run", action="store_true", help="只顯示結果，不做實際搬/複製")
    args = parser.parse_args()

    if args.move and args.copy:
        raise SystemExit("請只選一個：--copy 或 --move")
    do_move = bool(args.move)
    prefer = None if args.prefer == "none" else args.prefer

    src_root = Path(args.src)
    out_dog = Path(args.out_dog)
    out_cat = Path(args.out_cat)
    out_unknown = Path(args.out_unknown)

    splitter = SpeciesSplitter(args.yolo)

    # 統計
    stats = {
        "dog": 0,
        "cat": 0,
        "unknown": 0,
        "ambiguous": 0,
        "bad_image": 0,
        "total": 0,
    }

    items = list(iter_images_by_split(src_root))
    if not items:
        raise SystemExit(f"在 {src_root} 找不到 train/val 圖片資料夾結構（ImageFolder）。")

    pbar = tqdm(items, desc="Splitting by species")
    for split, label, img_path in pbar:
        stats["total"] += 1
        img = safe_open_image(img_path)
        if img is None:
            stats["bad_image"] += 1
            continue

        decision = splitter.decide_species(
            img,
            conf_thres=args.conf,
            ambiguous_margin=args.amb_margin,
            prefer=prefer,
        )

        target_root = None
        if decision.species == "dog":
            target_root = out_dog
            stats["dog"] += 1
        elif decision.species == "cat":
            target_root = out_cat
            stats["cat"] += 1
        elif decision.species == "ambiguous":
            target_root = out_unknown
            stats["ambiguous"] += 1
        else:
            target_root = out_unknown
            stats["unknown"] += 1

        dst_dir = target_root / split / label
        ensure_dir(dst_dir)
        dst_path = unique_name(dst_dir, img_path.name)

        pbar.set_postfix(species=decision.species, conf=round(decision.best_conf, 3))

        if args.dry_run:
            continue

        if do_move:
            shutil.move(str(img_path), str(dst_path))
        else:
            shutil.copy2(str(img_path), str(dst_path))

    print("\n=== Done ===")
    print(f"src: {src_root.resolve()}")
    print(f"out_dog: {out_dog.resolve()}")
    print(f"out_cat: {out_cat.resolve()}")
    print(f"out_unknown: {out_unknown.resolve()}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
