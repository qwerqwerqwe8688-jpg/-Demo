# -*- coding: utf-8 -*-
"""
data_fabric.py
--------------
递归扫描 datasets 文件夹（支持直接放图片的根目录），
生成统一索引 fabric_index.csv。
"""
import os
import csv
import argparse
from datetime import datetime

def is_image_file(filename):
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    return filename.lower().endswith(IMG_EXTS)

def scan_dataset(root_dir):
    """
    递归扫描 root_dir 下所有图片文件。
    """
    items = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if is_image_file(fname):
                fpath = os.path.join(dirpath, fname)
                try:
                    fstat = os.stat(fpath)
                    rel_dir = os.path.relpath(dirpath, root_dir)
                    # 如果图像在根目录下，使用 "root" 作为标签
                    label = os.path.basename(rel_dir) if rel_dir != "." else "root"
                    items.append({
                        "path": os.path.abspath(fpath),
                        "label": label,
                        "size_kb": round(fstat.st_size / 1024, 2),
                        "mtime": datetime.fromtimestamp(fstat.st_mtime).isoformat(timespec='seconds')
                    })
                except Exception as e:
                    print(f"[WARN] Skip file {fpath}: {e}")
    return items

def write_csv(items, out_csv):
    """
    输出 CSV 文件。
    """
    if not items:
        print("[WARN] No image found.")
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "size_kb", "mtime"])
        return

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "size_kb", "mtime"])
        writer.writeheader()
        for item in items:
            writer.writerow(item)
    print(f"[INFO] Wrote {len(items)} items to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fabric index for dataset.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root folder")
    parser.add_argument("--out", type=str, default="fabric_index.csv", help="Output CSV file")
    args = parser.parse_args()

    print(f"[INFO] Scanning dataset root: {args.root}")
    data_items = scan_dataset(args.root)
    write_csv(data_items, args.out)
