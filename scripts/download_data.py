import sys

sys.path.append('..')

import os
import kaggle
from typing import Union, Optional


def download_kaggle_dataset(
        dataset_name: str,
        # download_path: Union[str, None] = None,
        download_path: Optional[str] = None,
) -> None:
    """
    Downloads a dataset from Kaggle.  从 Kaggle 下载数据集。

    :param dataset_name: The name of the dataset on Kaggle.  Kaggle 数据集的名称。
    :param download_path: The local path where the dataset should be downloaded. If None, defaults to '../data/{dataset_name}'.  本地下载数据集的路径。如果为 None，则默认为 '../data/{dataset_name}'
    """
    try:
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.join('.keys')  # Set Kaggle config directory to keys folder  # 将 Kaggle 配置目录设置为 keys 文件夹
        kaggle.api.authenticate()  # Authenticate with Kaggle  # 使用 Kaggle 进行身份验证
    except Exception as e:
        print(f"An error occurred while setting KAGGLE_CONFIG_DIR: {e}")
        return
    try:
        if download_path is None or download_path.strip() == "":
            download_path = os.path.join(
                '../data',  # Parent 'data' directory  # 父级 'data' 目录
                dataset_name.replace('/', '_')  # Dataset name as folder name  # 数据集名称作为文件夹名称
            )
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset '{dataset_name}' downloaded successfully to '{download_path}'.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")


if __name__ == "__main__":
    dataset_name = "ibrahimfateen/wound-classification"  # Example dataset name  # 示例数据集名称
    download_path = '../data'

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    download_kaggle_dataset(
        dataset_name,
        download_path
    )
