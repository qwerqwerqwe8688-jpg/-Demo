# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

def train(index_csv, feats_file, model_file, n_clusters=5):
    # 读取特征
    X = np.load(feats_file)
    print(f"[INFO] Loaded features: {X.shape}")

    # 聚类
    print(f"[INFO] Training KMeans with {n_clusters} clusters...")
    clf = KMeans(n_clusters=n_clusters, random_state=42)
    clf.fit(X)

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
    parser = argparse.ArgumentParser(description="Train unsupervised classifier (KMeans) on image features")
    parser.add_argument("--index", type=str, required=True, help="CSV index file (any format, no label needed)")
    parser.add_argument("--feats", type=str, required=True, help="Feature .npy file")
    parser.add_argument("--model", type=str, required=True, help="Output model file (.joblib)")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for KMeans")
    args = parser.parse_args()

    train(args.index, args.feats, args.model, n_clusters=args.n_clusters)
