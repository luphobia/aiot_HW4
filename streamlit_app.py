import os
import json
import glob
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms, models


# ----------------------------
# Page + CSS (bigger uploader + buttons)
# ----------------------------
st.set_page_config(page_title="Pet Emotion Classifier", layout="centered")
st.title("🐾 寵物情緒/表情辨識 Demo（單模型版）")
st.caption("上傳圖片 →（可選）自動主體裁切 → 輸出四種情緒機率分布")

st.markdown(
    """
<style>
/* 放大 file_uploader */
div[data-testid="stFileUploader"] section {
    padding: 18px 14px;
}
div[data-testid="stFileUploader"] button {
    font-size: 18px !important;
    padding: 10px 18px !important;
}
div[data-testid="stFileUploader"] small {
    font-size: 14px !important;
}

/* 放大一般按鈕 */
div.stButton > button {
    font-size: 16px !important;
    padding: 10px 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Model loading (single model: pet_emotion_*.pt/json)
# ----------------------------
def load_latest_model(models_dir: str = "models", prefix: str = "pet_emotion"):
    pts = sorted(glob.glob(os.path.join(models_dir, f"{prefix}_*.pt")))
    metas = sorted(glob.glob(os.path.join(models_dir, f"{prefix}_*.json")))
    if not pts or not metas:
        return None, None, None, None, None

    model_path = pts[-1]
    meta_path = metas[-1]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    img_size = int(meta.get("img_size", 224))

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(classes))

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, classes, img_size, model_path, meta_path


def preprocess(img: Image.Image, img_size: int):
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return tf(img).unsqueeze(0)


@torch.no_grad()
def predict_all(model, classes, img: Image.Image, img_size: int):
    x = preprocess(img, img_size)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    return {classes[i]: float(probs[i]) for i in range(len(classes))}


def normalize_label(s: str) -> str:
    """
    把標籤統一成小寫，並簡單處理 other/others 這種常見差異
    """
    s2 = (s or "").strip().lower()
    if s2 == "others":
        return "other"
    return s2


# ----------------------------
# Load model
# ----------------------------
models_dir = "models"
model, classes, img_size, model_path, meta_path = load_latest_model(models_dir, prefix="pet_emotion")

if model is None:
    st.warning(
        "找不到 pet_emotion_* 模型。\n\n"
        "請先訓練單模型：\n"
        "- python train.py --data_dir data --tag pet_emotion\n"
    )
    st.stop()

# 固定顯示 4 類（依你的需求）
# 會用 normalize_label 對齊 classes 名稱
DISPLAY_CLASSES = ["happy", "sad", "angry", "other"]

# 顯示用：檢查模型實際 classes
with st.expander("🔎 模型資訊（classes / 檔案）", expanded=False):
    st.write("classes:", classes)
    st.write("model:", model_path)
    st.write("meta:", meta_path)
    missing = [c for c in DISPLAY_CLASSES if c not in [normalize_label(x) for x in classes]]
    if missing:
        st.warning(f"注意：模型 classes 內找不到以下類別：{missing}（可能命名不同或訓練資料夾名稱不一致）")


# ----------------------------
# Inputs: upload + sample
# ----------------------------
st.subheader("輸入圖片")

colA, colB = st.columns([2, 1])

with colA:
    uploaded = st.file_uploader("上傳寵物照片（jpg/png）", type=["jpg", "jpeg", "png", "webp"])

with colB:
    sample_files = sorted(
        glob.glob("samples/*.jpg")
        + glob.glob("samples/*.jpeg")
        + glob.glob("samples/*.png")
        + glob.glob("samples/*.webp")
    )
    sample_options = ["（不使用）"] + [os.path.basename(p) for p in sample_files]
    sample_name = st.selectbox("或選擇範例圖片", sample_options)
    use_sample = st.button("用範例圖片測試", use_container_width=True)

    # ROI 裁切選項：不顯示 ROI、不顯示 cat/dog，只當成內部前處理
    use_detect = st.checkbox("啟用自動主體裁切（建議）", value=True)
    conf_thres = st.slider("裁切偵測信心閾值", min_value=0.05, max_value=0.90, value=0.25, step=0.05)

img = None
img_source = None

if use_sample and sample_name != "（不使用）":
    sample_path = [p for p in sample_files if os.path.basename(p) == sample_name][0]
    img = Image.open(sample_path).convert("RGB")
    img_source = f"範例圖片：{sample_name}"
elif uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_source = "上傳圖片"

if img is None:
    st.info("請上傳圖片，或選擇範例圖片並按下「用範例圖片測試」。")
    st.stop()

# Fix EXIF rotation
img = ImageOps.exif_transpose(img)

st.image(img, caption=f"{img_source}", use_container_width=True)


# ----------------------------
# Optional detect + ROI crop (lazy import, cloud-safe)
# ----------------------------
roi_img = img

detector = None
if use_detect:
    try:
        # Lazy import: 避免部署環境因 cv2/ultralytics 直接掛掉
        from detector import PetDetector  # noqa: F401

        detector = PetDetector("yolov8n.pt")
    except Exception:
        # 不顯示錯誤細節（雲端常會 redacted），只做降級提示
        st.warning("自動主體裁切在目前部署環境無法啟用，已改用原圖進行推論。")
        detector = None
        use_detect = False

if use_detect and detector is not None:
    try:
        det = detector.detect_and_crop(img, conf_thres=conf_thres, pad_ratio=0.10)
        roi_img = det.crop
    except Exception:
        # 裁切失敗也要能回退，不影響主流程
        roi_img = img


# ----------------------------
# Predict: fixed 4 classes + full table + bar chart
# ----------------------------
raw_probs = predict_all(model, classes, roi_img, img_size)

# 將模型輸出映射到固定四類（以 normalize_label 對齊）
norm_map = {normalize_label(k): v for k, v in raw_probs.items()}

# 若模型本身類別命名不同（例如 Other/others），這裡會盡量對齊
fixed_items = []
for c in DISPLAY_CLASSES:
    fixed_items.append((c, float(norm_map.get(c, 0.0))))

# 依機率高到低排序顯示（但永遠顯示四類）
fixed_items_sorted = sorted(fixed_items, key=lambda x: x[1], reverse=True)

st.subheader("推論結果（四種情緒機率）")
for label, p in fixed_items_sorted:
    st.write(f"**{label}**：{p * 100:.2f}%")

st.divider()
st.caption("完整機率表與圖表（固定四類）")

df = pd.DataFrame(fixed_items_sorted, columns=["label", "prob"])
df["prob_%"] = df["prob"] * 100.0

st.dataframe(df[["label", "prob_%"]], use_container_width=True, hide_index=True)
st.bar_chart(df.set_index("label")[["prob_%"]])
