from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False


@dataclass
class DetResult:
    species: str
    conf: float
    crop: Image.Image
    bbox: Tuple[int, int, int, int]


class PetDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        if not _YOLO_OK:
            raise ImportError("ultralytics/cv2 無法載入（部署環境不支援）。請關閉 ROI 裁切或修正 requirements/runtime。")
        self.model = YOLO(model_name)

    def detect_and_crop(self, img: Image.Image, conf_thres: float = 0.25, pad_ratio: float = 0.10) -> DetResult:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        im_np = np.array(img_rgb)

        r = self.model.predict(im_np, conf=conf_thres, verbose=False)[0]
        best = None

        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                if cls_id not in (15, 16):  # cat=15 dog=16
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                area = max(1.0, (x2 - x1) * (y2 - y1))
                score = conf + 0.000001 * area
                if best is None or score > best[0]:
                    best = (score, cls_id, conf, x1, y1, x2, y2)

        if best is None:
            return DetResult("unknown", 0.0, img_rgb, (0, 0, w, h))

        _, cls_id, conf, x1, y1, x2, y2 = best
        species = "cat" if cls_id == 15 else "dog"

        pad_w = (x2 - x1) * pad_ratio
        pad_h = (y2 - y1) * pad_ratio
        nx1 = int(max(0, x1 - pad_w))
        ny1 = int(max(0, y1 - pad_h))
        nx2 = int(min(w, x2 + pad_w))
        ny2 = int(min(h, y2 + pad_h))

        crop = img_rgb.crop((nx1, ny1, nx2, ny2))
        return DetResult(species, conf, crop, (nx1, ny1, nx2, ny2))
