# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image
import tifffile

# ================= 图像安全加载函数 =================
def safe_load_image(path):
    """
    智能加载图像：
    - 普通 RGB 图片
    - 单通道 / 两通道 / 多通道 .tif 遥感 / SAR 图像
    - 异常文件自动跳过
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        img = None

        # 优先用 tifffile 读取
        if ext in [".tif", ".tiff"]:
            try:
                img = tifffile.imread(path)
            except Exception:
                pass

        # tifffile 失败或其他格式 fallback PIL
        if img is None:
            img = Image.open(path)
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'I;16':
                img = img.point(lambda i: i*(1./256)).convert('RGB')
            else:
                img = img.convert('RGB')
        else:
            # numpy array 处理通道
            if img.ndim == 2:  # 单通道
                img = np.stack([img]*3, axis=-1)
            elif img.ndim == 3:
                if img.shape[2] == 1:  # 单通道
                    img = np.concatenate([img]*3, axis=2)
                elif img.shape[2] == 2:  # 两通道
                    img = np.concatenate([img, img[:, :, 0:1]], axis=2)
                elif img.shape[2] > 3:  # 多通道
                    img = img[:, :, :3]
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

        return img
    except Exception as e:
        print(f"[ERROR] Cannot process {path}: {e}")
        return None

# ================= 特征提取函数 =================
def extract_features(index_csv, out_file, pretrained=True):
    df = pd.read_csv(index_csv)

    # 读取 path 列
    if "path" not in df.columns:
        raise ValueError("CSV must contain a 'path' column.")
    paths = df["path"].tolist()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载 ResNet50 模型（去掉最后分类层）
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])

    features, valid_paths = [], []

    for path in tqdm(paths):
        img = safe_load_image(path)
        if img is None:
            continue

        x = transform(img).unsqueeze(0)
        if x.shape[1] != 3:
            print(f"[WARN] Skipping {path}, unexpected channel count {x.shape[1]}")
            continue

        with torch.no_grad():
            feat = model(x).squeeze().numpy()

        features.append(feat)
        valid_paths.append(path)

    feats = np.array(features)
    np.save(out_file, feats)
    print(f"[INFO] Saved {len(feats)} feature vectors to {out_file}")

    df_valid = df[df["path"].isin(valid_paths)]
    df_valid.to_csv("fabric_index_valid.csv", index=False)
    print(f"[INFO] Saved valid index to fabric_index_valid.csv")

# ================= 主程序 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image features from dataset")
    parser.add_argument("--index", type=str, required=True, help="CSV index file (must contain 'path' column)")
    parser.add_argument("--out", type=str, required=True, help="Output .npy feature file")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights")
    args = parser.parse_args()

    extract_features(args.index, args.out, pretrained=args.pretrained)
