# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import random  # 新增随机选择功能


# ================= 加载资源 =================
def load_resources(index_csv, feats_npy, model_file):
    df = pd.read_csv(index_csv)
    feats = np.load(feats_npy)
    model = joblib.load(model_file)  # KMeans对象
    return df, feats, model


# ================= UMAP 可视化 =================
def plot_umap(feats, cluster_labels):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(feats)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=cluster_labels, palette='tab10', s=30, legend='full'
    )
    plt.title("UMAP projection of clustered images")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    return buf


# ================= 根据簇显示样本图像（随机4x4网格） =================
def show_cluster_images(cluster_idx, df, cluster_labels, max_images=16):
    # 筛选簇内所有图像路径
    paths = df['path'][cluster_labels == cluster_idx].tolist()
    if not paths:
        return None
    # 随机选择最多 max_images 张图
    selected_paths = random.sample(paths, min(max_images, len(paths)))

    imgs = []
    for p in selected_paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except:
            continue
    if not imgs:
        return None

    # 构建 4x4 网格
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid


# ================= 构建 Gradio 界面 =================
def build_ui(index_csv, feats_npy, model_file):
    df, feats, model = load_resources(index_csv, feats_npy, model_file)
    cluster_labels = model.labels_
    cluster_options = sorted(list(set(cluster_labels)))
    cluster_options_str = [str(c) for c in cluster_options]

    umap_buf = plot_umap(feats, cluster_labels)

    with gr.Blocks() as demo:
        gr.Markdown("## 多源异构数据编织Demo（以应用于无监督图像聚类可视化任务为例）")
        with gr.Row():
            umap_img = gr.Image(value=Image.open(umap_buf), label="UMAP Visualization", type="pil")
            with gr.Column():
                cluster_select = gr.Dropdown(label="Select Cluster", choices=cluster_options_str,
                                             value=cluster_options_str[0])
                cluster_img = gr.Image(label="Cluster Sample Images", type="pil")

        # 回调函数必须在 Blocks 内部定义
        def update_cluster(cluster_idx):
            return show_cluster_images(int(cluster_idx), df, cluster_labels)

        cluster_select.change(fn=update_cluster, inputs=cluster_select, outputs=cluster_img)

    return demo


# ================= 主程序 =================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gradio App for Unsupervised Image Clustering")
    parser.add_argument("--index", type=str, required=True, help="CSV index file")
    parser.add_argument("--feats", type=str, required=True, help="Feature .npy file")
    parser.add_argument("--model", type=str, required=True, help="KMeans model file (.joblib)")
    args = parser.parse_args()

    demo = build_ui(args.index, args.feats, args.model)
    demo.launch(server_name="127.0.0.1", server_port=7860)
