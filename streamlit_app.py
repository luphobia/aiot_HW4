import os
import json
import glob
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms, models

from detector import PetDetector


# ----------------------------
# Page + CSS (bigger uploader + buttons)
# ----------------------------
st.set_page_config(page_title="Pet Emotion Classifier", layout="centered")
st.title("🐾 寵物情緒/表情辨識 Demo（單模型版）")
st.caption("流程：可選偵測裁切 ROI → 單一情緒模型推論 → 顯示 Top-K + 全部類別機率")

st.markdown("""
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
""", unsafe_allow_html=True)


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
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def predict_all(model, classes, img: Image.Image, img_size: int):
    x = preprocess(img, img_size)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    return {classes[i]: float(probs[i]) for i in range(len(classes))}


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

with st.expander("🔎 目前載入的模型資訊（classes / 檔案）", expanded=False):
    st.write("classes:", classes)
    st.write("model:", model_path)
    st.write("meta:", meta_path)

# Detector (YOLO) - 用來裁切 ROI（可關閉）
detector = PetDetector("yolov8n.pt")


# ----------------------------
# Inputs: upload + sample
# ----------------------------
st.subheader("輸入圖片")

colA, colB = st.columns([2, 1])

with colA:
    uploaded = st.file_uploader("上傳寵物照片（jpg/png）", type=["jpg", "jpeg", "png"])

with colB:
    sample_files = sorted(
        glob.glob("samples/*.jpg") +
        glob.glob("samples/*.jpeg") +
        glob.glob("samples/*.png") +
        glob.glob("samples/*.webp")
    )
    sample_options = ["（不使用）"] + [os.path.basename(p) for p in sample_files]
    sample_name = st.selectbox("或選擇範例圖片", sample_options)

    use_sample = st.button("用範例圖片測試", use_container_width=True)
    # show_roi = st.checkbox("顯示裁切 ROI", value=True)
    use_detect = st.checkbox("啟用偵測與 ROI 裁切（建議）", value=True)

# topk = st.slider("顯示 Top-K", min_value=1, max_value=min(10, len(classes)), value=min(3, len(classes)))
conf_thres = st.slider("偵測信心閾值", min_value=0.05, max_value=0.90, value=0.25, step=0.05)

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
st.image(img, caption=f"{img_source}（原始輸入圖片）", use_container_width=True)


# ----------------------------
# Optional detect + ROI crop
# ----------------------------
roi_img = img
if use_detect:
    det = detector.detect_and_crop(img, conf_thres=conf_thres, pad_ratio=0.10)
    roi_img = det.crop


# ----------------------------
# Predict: TopK + full probabilities
# ----------------------------
probs_dict = predict_all(model, classes, roi_img, img_size)
items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

st.subheader("推論結果（四種情緒機率）")
for label, p in items:
    st.write(f"**{label}**：{p*100:.2f}%")


st.divider()
st.caption("完整類別機率（查看所有類別機率分布）")

df = pd.DataFrame(items, columns=["label", "prob"])
df["prob_%"] = df["prob"] * 100.0
st.dataframe(df[["label", "prob_%"]], use_container_width=True, hide_index=True)

st.bar_chart(df.set_index("label")[["prob_%"]])
