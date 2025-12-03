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
        save_path: Optional[str] = '../data/dataset',
):
    """
    Cleans the dataset at the given path by removing corrupted images.  通过删除损坏的图像来清理给定路径下的数据集。
    At the same time, not copy images starting with 'mirrored_'.  同时，不复制以 'mirrored_' 开头的图像。

    :param data_path: The local path where the dataset is stored. If None, defaults to '../data/Wound_dataset copy'.  本地存储数据集的路径。如果为 None，则默认为 '../data/Wound_dataset copy'。
    :param save_path: The local path where the cleaned dataset should be saved. If None, defaults to '../data/dataset'.  清理后的数据集应保存的本地路径。如果为 None，则默认为 '../data/dataset'。
    """
    if data_path is None or data_path.strip() == "":  # Default data path  # 默认数据路径
        data_path = os.path.join('../data/Wound_dataset copy')
    if save_path is None or save_path.strip() == "":  # Default save path  # 默认保存路径
        save_path = os.path.join('../data/dataset')
    save_path = save_path.strip()  # Remove leading/trailing whitespace  # 删除前导/尾随空格
    if os.path.exists(save_path):  # If cleaned dataset directory exists  # 如果清理后的数据集目录存在
        shutil.rmtree(save_path)  # Remove existing cleaned dataset directory  # 删除现有的清理后数据集目录
    os.makedirs(save_path)  # Create cleaned dataset directory  # 创建清理后的数据集目录
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
                    not fname.lower().endswith(('.png', '.jpg', '.jpeg'))  # Process only image files  # 仅处理图像文件
                    or fname.lower().startswith('mirrored_')
            ):  # Process only .jpg files and skip mirrored images  # 仅处理 .jpg 文件并跳过镜像图像
                continue
            try:
                fpath = os.path.join(label_path, fname)  # Full file path  # 完整文件路径
                safe_label_name = label.replace('/', '_')  # Safe label name for directory creation  # 用于目录创建的安全标签名称
                dest_label_path = os.path.join(
                    save_path,  # Destination cleaned dataset directory  # 目标清理后数据集目录
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
    print(f"[Info] Corrupted images: {corrupted_images}") if corrupted_images else print("[Info] No corrupted images found.")
    print(
        f"[Info] Total images processed: {index - 1} "
        f" (excluding corrupted images: {len(corrupted_images)})"
    ) if corrupted_images else f"[Info] Total images processed: {index - 1}"


def build_train_and_test_split(
        data_path: Optional[str] = None,
        save_path: Optional[str] = None,
        train_split_ratio: float = 0.9,
):
    """
    Builds train and test splits from the cleaned dataset.  从清理后的数据集中构建训练和测试集划分。

    :param data_path: The local path where the cleaned dataset is stored. If None, defaults to '../data/dataset'.  本地存储清理后数据集的路径。如果为 None，则默认为 '../data/dataset'。
    :param save_path: The local path where the train and test splits should be saved. If None, defaults to '../data/dataset_split'.  训练和测试集划分应保存的本地路径。如果为 None，则默认为 '../data/dataset_split'。
    :param train_split_ratio: The ratio of the dataset to be used for training. The rest will be used for testing.  用于训练的数据集比例。其余部分将用于测试。
    """
    from scripts.dataset import load_dataset
    if data_path is None or data_path.strip() == "":
        data_path = os.path.join('../data/dataset')
    if save_path is None or save_path.strip() == "":
        save_path = os.path.join('../data/dataset_split')
    if os.path.exists(save_path):  # If dataset split directory exists  # 如果数据集划分目录存在
        shutil.rmtree(save_path)  # Remove existing dataset split directory  # 删除现有的数据集划分目录
    os.makedirs(save_path)  # Create dataset split directory  # 创建数据集划分
    dataset_train, dataset_test, unique_labels = load_dataset(
        data_path=data_path,
        train_split_ratio=train_split_ratio,
    )
    print(f"[Info] Unique labels: {unique_labels}")
    print(f"[Info] Total training samples: {len(dataset_train)}")
    print(f"[Info] Total testing samples: {len(dataset_test)}")
    # Copy the files to the new directory  # 将文件复制到新目录
    train_save_path = os.path.join(save_path, 'train')  # Create train directory  # 创建训练目录
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'test')  # Create test directory  # 创建测试目录
    os.makedirs(test_save_path, exist_ok=True)
    for item in dataset_train:
        fname = os.path.basename(item['path'])
        dest_path = os.path.join(train_save_path, fname)
        shutil.copyfile(item['path'], dest_path)
    for item in dataset_test:
        fname = os.path.basename(item['path'])
        dest_path = os.path.join(test_save_path, fname)
        shutil.copyfile(item['path'], dest_path)
    print(f"[Info] Train and test splits saved to '{save_path}'.")

if __name__ == "__main__":
    data_path = os.path.join('../data/Wound_dataset copy')
    print("=" * 20 + " Inspecting Data " + "=" * 20)
    try:
        inspect_data(data_path)
    except Exception as e:
        print(f"[Error] An error occurred during data inspection: {e}")
    print("=" * 20 + " Cleaning Data " + "=" * 20)
    try:
        clean_data(
            data_path,
            save_path='../data/dataset',
        )
    except Exception as e:
        print(f"[Error] An error occurred during data cleaning: {e}")
    print("=" * 20 + " Building Train and Test Split " + "=" * 20)
    try:
        build_train_and_test_split(
            data_path='../data/dataset',
            save_path='../data/dataset_split',
            train_split_ratio=0.9,
        )
    except Exception as e:
        print(f"[Error] An error occurred during train/test split building: {e}")
    print("=" * 20 + " End " + "=" * 20)
