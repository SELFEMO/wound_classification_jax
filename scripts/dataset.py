import sys

sys.path.append('..')

import os
from typing import Optional, List

import numpy
import PIL
from PIL import Image, ImageEnhance, ImageFilter


def fname_to_index_and_label(
        fname: str
) -> Optional[tuple]:
    """
    Extracts the index and label from the filename.  从文件名中提取索引和标签。
    Expected filename format: 'index_label.ext'  预期的文件名格式：'index_label.ext'

    :param fname: The filename to extract information from.  要从中提取信息的文件名。

    :return: A tuple of (index, label) if extraction is successful, otherwise None.  如果提取成功，则返回 (index, label) 元组，否则返回 None。
    """
    try:
        base_name = os.path.splitext(fname)[0]  # Remove file extension  去除文件扩展名
        index_str, label = base_name.split('_')  # Split by underscore  按下划线分割
        index = int(index_str)  # Convert index to integer  将索引转换为整数
        return index, label
    except Exception as e:
        print(f"[Error] Could not parse filename {fname}: {e}")
        return None


def load_dataset(
        data_path: Optional[str] = None,
        train_split_ratio: Optional[float] = 1.0,
) -> Optional[tuple]:
    """
    Loads and preprocesses the dataset from the given path.  从给定路径加载和预处理数据集。

    :param data_path: The local path where the dataset is stored. If None, returns None.  本地存储数据集的路径。如果为 None，则返回 None。
    :param train_split_ratio: The proportion of the dataset to be used for training. If 1.0, the entire dataset is used for training.  用于训练的数据集比例。如果为 1.0，则整个数据集用于训练。

    :return: A tuple of (dataset_train, dataset_test, unique_labels) if successful, otherwise None.  如果成功，则返回 (dataset_train, dataset_test, unique_labels) 元组，否则返回 None。
    """
    if data_path is None or data_path.strip() == "":
        return None

    dataset = []  # List to hold dataset entries  # 用于保存数据集条目的列表
    all_labels = []  # List to hold all labels  # 用于保存所有标签的列表

    for fname in os.listdir(data_path):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        fpath = os.path.join(data_path, fname)
        if not os.path.isfile(fpath):
            continue
        index, label = fname_to_index_and_label(fname)
        if index is None or label is None:
            print(f"[Warning] Filename format incorrect: {fpath}")
            continue
        dataset.append({
            "index": index,
            "label_str": label,
            "path": fpath,
        })  # Store dataset entry  # 存储数据集条目
        all_labels.append(label)  # Add label to all_labels list  # 将标签添加到 all_labels 列表

    unique_labels = sorted(set(all_labels))  # Get unique labels sorted  # 获取唯一标签并排序

    group_dataset = {label_name: [] for label_name in unique_labels}  # Initialize grouped dataset  # 初始化分组数据集
    for data in dataset:
        group_dataset[data['label_str']].append({
            "index": data['index'],
            "label_str": data['label_str'],
            "label_index": unique_labels.index(data['label_str']),
            "path": data['path'],
        })

    dataset_train, dataset_test = [], []  # Lists to hold train and test datasets  # 用于保存训练和测试数据集的列表
    for label_name, items in group_dataset.items():  # For each label group  # 对于每个标签组
        # Shuffle items for randomness  # 打乱条目以增加随机性
        numpy.random.shuffle(items)
        # Calculate split index  # 计算分割索引
        split_idx = int(len(items) * train_split_ratio)
        # Add to train and test datasets  # 添加到训练和测试数据集
        dataset_train.extend(items[:split_idx])
        dataset_test.extend(items[split_idx:])

    # Shuffle the final datasets  # 打乱最终数据集
    numpy.random.shuffle(dataset_train)
    numpy.random.shuffle(dataset_test)

    return dataset_train, dataset_test, unique_labels


def apply_augmentation(
        image: Image.Image,
) -> Image.Image:
    """
    Applies simple augmentation techniques to the given image.  对给定图像应用简单的增强技术。

    :param image: The input PIL Image to augment.  要增强的输入 PIL 图像。

    :return: The augmented image as a numpy array.  增强后的图像作为 numpy 数组。
    """
    # Store original size  存储原始大小
    w, h = image.size

    # is_special_augmentation = False
    is_special_augmentation = numpy.random.choice([True, False])

    # Horizontal flip  水平翻转
    if numpy.random.rand() > 0.3:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip  水平翻转

    # # Vertical flip 垂直翻转
    # if numpy.random.rand() > 0.4:
    #     image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip  垂直翻转

    # Random rotation  随机旋转
    if numpy.random.rand() > 0.7 and not is_special_augmentation:
        # angle = numpy.random.uniform(-360, 360)
        angle = numpy.random.uniform(-10, 10)
        image = image.rotate(angle)
        is_special_augmentation = True

    # Random brightness adjustment  随机亮度调整
    if numpy.random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(numpy.random.uniform(0.9, 1.1))

    # Random contrast adjustment  随机对比度调整
    if numpy.random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(numpy.random.uniform(0.9, 1.1))

    # Random zoom  随机缩放
    if numpy.random.rand() > 0.7 and not is_special_augmentation:
        zoom_factor = numpy.random.uniform(0.9, 1.1)
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        image = image.crop((left, top, left + w, top + h))
        image = image.resize((w, h), Image.Resampling.LANCZOS)  # Resize back to original size  调整回原始大小
        is_special_augmentation = True

    # Random color jitter  随机颜色抖动
    if numpy.random.random() > 0.8:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(numpy.random.uniform(0.9, 1.1))

    # Gaussian noise  高斯噪声
    if numpy.random.random() > 0.9 and not is_special_augmentation:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=numpy.random.uniform(0.5, 1.5)
            )
        )

    return image


class data_loader:
    """
    Data loader class for loading and preprocessing images from a dataset.  用于从数据集中加载和预处理图像的数据加载器类。

    The dataset is expected to be organized in the following structure:  期望数据集组织结构如下：
    wound_classification_jax/
        data/
            dataset/
                000001_ClassA.jpg
                000002_ClassB.jpg
        ...
    """

    def __init__(
            self,
            data_path: Optional[str] = None,
            image_size: Optional[tuple] = (224, 224),
            train_split_ratio: Optional[float] = 0.9,
            use_augmentation: Optional[bool] = False,
    ):
        """
        Initializes the data loader with the given parameters.  使用给定的参数初始化数据加载器。

        :param data_path: The local path where the dataset is stored. If None, defaults to '../data/dataset'.  本地存储数据集的路径。如果为 None，则默认为 '../data/dataset'。
        :param image_size: The desired size to which each image will be resized. If None, images will be resized to (224, 224).  每个图像将调整为的所需大小。如果为 None，则图像将调整为 (224, 224)。
        """
        if data_path is None or data_path.strip() == "":
            data_path = os.path.join('../data/dataset')
        self.data_path = data_path
        self.image_size = image_size if image_size is not None else (224, 224)
        self.train_split_ratio = train_split_ratio if train_split_ratio is not None else 0.9
        self.dataset_train, self.dataset_test, self.unique_labels = load_dataset(
            data_path=self.data_path,
            image_size=self.image_size,
            train_split_ratio=self.train_split_ratio,
        )
        self.num_classes = len(self.unique_labels)
        self.use_augmentation = use_augmentation
        self.is_last = False
        self.is_training = True
        self.augmentation = apply_augmentation

    def __getitem__(
            self,
            index: int,
    ) -> Optional[tuple]:
        """
        Retrieves and preprocesses the image and label at the given index from the training dataset.  从训练数据集中检索并预处理给定索引处的图像和标签。

        :param index: The index of the data item to retrieve.  要检索的数据项的索引。

        :return: A tuple of (image, label_index, image_index) if successful, otherwise None.  如果成功，则返回 (image, label_index, image_index) 元组，否则返回 None。
        """
        # Check index bounds  检查索引范围
        if self.is_training and (index < 0 or index >= len(self.dataset_train)):
            return None
        elif not self.is_training and (index < 0 or index >= len(self.dataset_test)):
            return None

        # Retrieve data entry  检索数据条目
        data = self.dataset_train[index] if self.is_training else self.dataset_test[index]

        # Load and preprocess image  加载和预处理图像
        try:
            with Image.open(data['path']) as image:
                image = image.convert('RGB')  # Ensure image is in RGB format  确保图像为 RGB 格式
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)  # Resize image  调整图像大小
        except Exception as e:
            print(f"[Error] Could not load image {data['path']}: {e}")
            return None

        if self.use_augmentation and not self.is_last:
            image = apply_augmentation(image)

        image = numpy.array(image) / 255.0  # Normalize pixel values to [0, 1]  将像素值归一化到 [0, 1]
        # image = numpy.array(image) / 255.0 - 0.5  # Normalize pixel values to [-0.5, 0.5]  将像素值归一化到 [-0.5, 0.5]  # BUG: this normalization seems to cause issues

        return image, data['label_index'], data['index']

    def __len__(
            self,
    ) -> tuple[int, int]:
        """
        Returns the length of the training dataset.  返回训练数据集的长度。

        :return: A tuple of (train_length, test_length).  (train_length, test_length) 元组。
        """
        return len(self.dataset_train), len(self.dataset_test)

    def get_train_dateset(
            self,
    ) -> list:
        """
        Returns the training dataset.  返回训练数据集。

        :return: The training dataset as a list.  训练数据集作为列表。
        """
        return self.dataset_train

    def get_test_dateset(
            self,
    ) -> list:
        """
        Returns the testing dataset.  返回测试数据集。

        :return: The testing dataset as a list.  测试数据集作为列表。
        """
        return self.dataset_test

    def set_training(
            self,
            is_training: bool,
    ) -> None:
        """
        Sets whether the data loader is in training mode.  设置数据加载器是否处于训练模式。

        :param is_training: True if in training mode, otherwise False.  如果处于训练模式，则为 True，否则为 False。
        """
        self.is_training = is_training

    def get_is_training(
            self,
    ) -> bool:
        """
        Returns whether the data loader is in training mode.  返回数据加载器是否处于训练模式。

        :return: True if in training mode, otherwise False.  如果处于训练模式，则为 True，否则为 False。
        """
        return self.is_training

    def set_last(
            self,
            is_last: bool,
    ) -> None:
        """
        Sets whether the current batch is the last batch.  设置当前批次是否为最后一个批次。

        :param is_last: True if the current batch is the last batch, otherwise False.  如果当前批次是最后一个批次，则为 True，否则为 False。
        """
        self.is_last = is_last

    def get_is_last(
            self,
    ) -> bool:
        """
        Returns whether the current batch is the last batch.  返回当前批次是否为最后一个批次。

        :return: True if the current batch is the last batch, otherwise False.  如果当前批次是最后一个批次，则为 True，否则为 False。
        """
        return self.is_last

    def get_image_size(
            self,
    ) -> tuple:
        """
        Returns the image size used for resizing.  返回用于调整大小的图像尺寸。

        :return: The image size as a tuple (width, height).  图像尺寸作为元组 (宽度, 高度)。
        """
        return self.image_size

    def get_num_classes(
            self,
    ) -> int:
        """
        Returns the number of unique classes in the dataset.  返回数据集中唯一类别的数量。

        :return: The number of unique classes.  唯一类别的数量。
        """
        return self.num_classes

    def get_unique_labels(
            self,
    ) -> List[str]:
        """
        Returns the list of unique labels in the dataset.  返回数据集中唯一标签的列表。

        :return: A list of unique label strings.  唯一标签字符串的列表。
        """
        return self.unique_labels


# ===== Test Code =====
if __name__ == "__main__":
    # Initialize data loader  初始化数据加载器
    loader = data_loader(
        data_path='../data/dataset',
        image_size=(224, 224),
        train_split_ratio=0.8,
        use_augmentation=True,
    )

    # Print dataset information  打印数据集信息
    print(f"Number of classes: {loader.get_num_classes()}")
    print(f"Unique labels: {loader.get_unique_labels()}")
    print(f"Training and validation dataset sizes: {loader.__len__()}")

    # Retrieve and print a sample data item  检索并打印一个样本数据项
    # sample_index = 0
    for sample_index in range(5):
        sample = loader.__getitem__(sample_index)
        if sample is not None:
            image, label_index, image_index = sample
            print(f"Sample index: {sample_index}, Image index: {image_index}, Label index: {label_index}, Image shape: {image.shape}")
        else:
            print(f"Could not retrieve sample at index {sample_index}")
