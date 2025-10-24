# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def auto_kmeans(X, min_k=2, max_k=10):
    """
    自动选择簇数：
    - 使用轮廓系数（Silhouette Score）评估每个 K
    - 返回最佳 K 和训练好的模型
    """
    best_k = min_k
    best_score = -1
    best_model = None

    for k in range(min_k, min(max_k+1, X.shape[0])):  # K不能超过样本数
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue  # 忽略只有1类的情况
        score = silhouette_score(X, labels)
        print(f"[INFO] K={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    print(f"[INFO] Selected best K={best_k} with silhouette score {best_score:.4f}")
    return best_model, best_k

def train(index_csv, feats_file, model_file, min_k=2, max_k=10):
    # 读取特征
    X = np.load(feats_file)
    print(f"[INFO] Loaded features: {X.shape}")

    # 自动聚类
    clf, best_k = auto_kmeans(X)

    # 保存模型
    joblib.dump(clf, model_file)
    print(f"[INFO] Saved clustering model to {model_file}")

    # 保存每个样本对应的簇标签到 CSV
    df = pd.read_csv(index_csv)
    if len(df) != X.shape[0]:
        print("[WARN] Index CSV length differs from feature count. Skipping label save.")
        return

    df['cluster_label'] = clf.labels_
    df.to_csv("fabric_index_clustered.csv", index=False)
    print("[INFO] Saved clustered labels to fabric_index_clustered.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unsupervised classifier (KMeans) with automatic cluster selection")
    parser.add_argument("--index", type=str, required=True, help="CSV index file (any format, no label needed)")
    parser.add_argument("--feats", type=str, required=True, help="Feature .npy file")
    parser.add_argument("--model", type=str, required=True, help="Output model file (.joblib)")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum number of clusters")
    args = parser.parse_args()

    train(args.index, args.feats, args.model, min_k=args.min_k, max_k=args.max_k)
