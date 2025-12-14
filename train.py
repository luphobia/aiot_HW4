from prepare_data import download_and_prepare
import os
import json
import time
import copy
from dataclasses import dataclass

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


@dataclass
class Config:
    data_dir: str = "data"
    train_dir: str = "train"
    val_dir: str = "val"
    out_dir: str = "models"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 8
    lr: float = 3e-4
    num_workers: int = 2
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(cfg.data_dir, cfg.train_dir)
    val_path = os.path.join(cfg.data_dir, cfg.val_dir)

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one(cfg: Config, tag: str = "pet_emotion"):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 如果 data/train 或 data/val 不存在，就自動下載+整理
    download_and_prepare(
        dataset_slug="anshtanwar/pets-facial-expression-dataset",
        out_dir=cfg.data_dir,
        val_ratio=0.15,
        seed=cfg.seed
    )

    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / max(len(pbar), 1))

        val_acc = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(cfg.out_dir, f"{tag}_{ts}.pt")
    meta_path  = os.path.join(cfg.out_dir, f"{tag}_{ts}.json")


    torch.save(best_state, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "img_size": cfg.img_size}, f, ensure_ascii=False, indent=2)

    print("Saved:", model_path)
    print("Saved:", meta_path)
    print("Best Val Acc:", best_acc)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tag", type=str, default="pet_emotion")  # 用來命名輸出檔
    args = parser.parse_args()

    cfg = Config()
    cfg.data_dir = args.data_dir

    # 在 train_one 裡存檔時，把檔名 pet_emotion 改成 cfg.tag 或 args.tag
    # 你可以把 train_one(cfg) 參數改成 train_one(cfg, tag=args.tag)
    # 我下面給你更快的改法：直接在 Config 加 tag 欄位也行
    train_one(cfg, tag=args.tag)

