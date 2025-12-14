from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# YOLOv8
from ultralytics import YOLO


@dataclass
class DetResult:
    species: str            # "dog" / "cat" / "unknown"
    conf: float
    crop: Image.Image       # 裁切後 ROI（PIL Image）
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 in original image


class PetDetector:
    """
    用 YOLOv8 COCO model 偵測 dog/cat，再把 ROI 裁切出來。
    COCO 類別 id：cat=15, dog=16（YOLOv8 預設）
    """
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect_and_crop(
        self,
        img: Image.Image,
        conf_thres: float = 0.25,
        pad_ratio: float = 0.10,   # bbox 外擴比例，讓 ROI 不會太緊
        prefer: Optional[str] = None,  # "dog"/"cat" or None
    ) -> DetResult:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        im_np = np.array(img_rgb)

        r = self.model.predict(im_np, conf=conf_thres, verbose=False)[0]

        best = None  # (score, cls_id, conf, x1,y1,x2,y2)
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                # 只要 dog/cat
                if cls_id not in (15, 16):
                    continue

                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

                species = "cat" if cls_id == 15 else "dog"

                # 如果指定 prefer，偏好該物種
                species_bonus = 0.05 if (prefer is not None and species == prefer) else 0.0

                # score：置信度 + 面積加權（偏好較大目標，通常是主體）
                area = max(1.0, (x2 - x1) * (y2 - y1))
                score = conf + 0.000001 * area + species_bonus

                if best is None or score > best[0]:
                    best = (score, cls_id, conf, x1, y1, x2, y2)

        if best is None:
            # 沒偵測到就回傳原圖
            return DetResult(
                species="unknown",
                conf=0.0,
                crop=img_rgb,
                bbox=(0, 0, w, h),
            )

        _, cls_id, conf, x1, y1, x2, y2 = best
        species = "cat" if cls_id == 15 else "dog"

        # bbox 外擴
        pad_w = (x2 - x1) * pad_ratio
        pad_h = (y2 - y1) * pad_ratio
        nx1 = int(max(0, x1 - pad_w))
        ny1 = int(max(0, y1 - pad_h))
        nx2 = int(min(w, x2 + pad_w))
        ny2 = int(min(h, y2 + pad_h))

        crop = img_rgb.crop((nx1, ny1, nx2, ny2))
        return DetResult(species=species, conf=conf, crop=crop, bbox=(nx1, ny1, nx2, ny2))
