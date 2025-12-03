import sys

sys.path.append('..')

import os
import shutil
import PIL
from typing import Optional


def inspect_data(
        data_path: Optional[str] = None,
) -> None:
    """
    Inspects the dataset at the given path and prints out basic information about the images.  检查给定路径下的数据集并打印图像的基本信息。

    :param data_path: The local path where the dataset is stored. If None, defaults to '../data/Wound_dataset copy'.  本地存储数据集的路径。如果为 None，则默认为 '../data/Wound_dataset copy'。
    """
    label_counts, size_counts, image_info = {}, {}, []  # Dictionaries to hold counts and list for image info  # 用于保存计数的字典和用于图像信息的列表
    if data_path is None or data_path.strip() == "":  # Default data path  # 默认数据路径
        data_path = os.path.join('../data/Wound_dataset copy')
    for label in os.listdir(data_path):
        label_path = os.path.join(
            data_path, label  # Path to the label directory  # 标签目录的路径
        )
        if not os.path.isdir(label_path):  # Ensure it's a directory  # 确保它是一个目录
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith('.jpg'):  # Process only .jpg files  # 仅处理 .jpg 文件
                continue
            fpath = os.path.join(label_path, fname)  # Full file path  # 完整文件路径
            try:
                with PIL.Image.open(fpath) as img:
                    w, h = img.size  # Get image dimensions  # 获取图像尺寸
            except Exception as e:
                print(f"[Error] Could not open image {fpath}: {e}")
                continue
            if label not in label_counts:  # Initialize label count  # 初始化标签计数
                label_counts[label] = 0
            label_counts[label] += 1  # Increment label count  # 增加标签计数
            # size_key = f"{w}x{h}"  # Size key as 'widthxheight'  # 尺寸键为 '宽度x高度'
            size_key = (w, h)  # Size key as (width, height) tuple  # 尺寸键为 (宽度, 高度) 元组
            if size_key not in size_counts:  # Initialize size count  # 初始化尺寸计数
                size_counts[size_key] = 0
            size_counts[size_key] += 1  # Increment size count  # 增加尺寸计数
            image_info.append({
                "path": fpath,
                "label": label,
                "width": w,
                "height": h,
            })  # Store image info  # 存储图像信息
    print(f"[Info] Label counts: {label_counts}")
    print(f"[Info] Image size counts: {size_counts}")
    print(f"[Info] Total images processed: {len(image_info)}")


def clean_data(
        data_path: Optional[str] = None,
):
    """
    Cleans the dataset at the given path by removing corrupted images.  通过删除损坏的图像来清理给定路径下的数据集。
    At the same time, not copy images starting with 'mirrored_'.  同时，不复制以 'mirrored_' 开头的图像。

    :param data_path: The local path where the dataset is stored. If None, defaults to '../data/Wound_dataset copy'.  本地存储数据集的路径。如果为 None，则默认为 '../data/Wound_dataset copy'。
    """
    if data_path is None or data_path.strip() == "":  # Default data path  # 默认数据路径
        data_path = os.path.join('../data/Wound_dataset copy')
    if os.path.exists('../data/dataset'):  # If cleaned dataset directory exists  # 如果清理后的数据集目录存在
        shutil.rmtree('../data/dataset')  # Remove existing cleaned dataset directory  # 删除现有的清理后数据集目录
    os.makedirs('../data/dataset')  # Create cleaned dataset directory  # 创建清理后的数据集目录
    corrupted_images = []  # List to hold paths of corrupted images  # 用于保存损坏图像路径的列表
    index = 1  # Index for renaming files  # 用于重命名文件的索引
    for label in os.listdir(data_path):
        label_path = os.path.join(
            data_path,  # Path to the label directory  # 标签目录的路径
            label  # 标签
        )
        if not os.path.isdir(label_path):  # Ensure it's a directory  # 确保它是一个目录
            continue
        for fname in os.listdir(label_path):
            if (
                    not fname.lower().endswith('.jpg')
                    or fname.startswith('mirrored_')
            ):  # Process only .jpg files and skip mirrored images  # 仅处理 .jpg 文件并跳过镜像图像
                continue
            try:
                fpath = os.path.join(label_path, fname)  # Full file path  # 完整文件路径
                safe_label_name = label.replace('/', '_')  # Safe label name for directory creation  # 用于目录创建的安全标签名称
                dest_label_path = os.path.join(
                    '../data/dataset',  # Destination cleaned dataset directory  # 目标清理后数据集目录
                    f'{index:06d}_{safe_label_name}.jpg'  # New filename with index and label  # 带有索引和标签的新文件名
                )
                with open(fpath, 'rb') as img:
                    data = img.read()
                with open(dest_label_path, 'wb') as dest_label:
                    dest_label.write(data)
            except Exception as e:
                print(f"[Error] Could not copy image {fpath}: {e}")
                corrupted_images.append(fpath)
                continue
            index += 1


if __name__ == "__main__":
    data_path = os.path.join('../data/Wound_dataset copy')

    inspect_data(data_path)

    clean_data(data_path)
