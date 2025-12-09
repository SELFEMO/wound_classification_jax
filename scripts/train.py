import os
import sys
import argparse
from typing import Tuple, Iterable, Optional, Any
import time
import pickle

from flax.core.nn import dropout
from tqdm import tqdm

import numpy as np
import jax
import flax
import optax
from PIL import Image

from flax.training import train_state

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from scripts.dataset import data_loader

from nets.CNN import SimpleCNN
from nets.ResNet import ResNet18, ResNet34
# from nets.Mamba import VisionMamba
from nets.Mamba import VisionMamba as Mamba

from nets.BaselineCNN import BaselineCNN
from nets.VisionMamba import VisionMamba
from nets.Hybrid import HybridMambaCNN, HybridMambaResNet


# ========================================
# Simple Checkpoint Tool (using pickle)
# 简易 Checkpoint 工具（使用 pickle）
# ========================================

def save_checkpoint(
        ckpt_path: str,
        data: dict
) -> None:
    """
    Saves a checkpoint to the specified path.  将检查点保存到指定路径。

    :param ckpt_path: The path where the checkpoint will be saved.  检查点将保存的路径。
    :param data: The data to be saved in the checkpoint.  要保存在检查点中的数据。
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(
        ckpt_path: str
) -> dict:
    """
    Loads a checkpoint from the specified path.  从指定路径加载检查点。

    :param ckpt_path: The path from where the checkpoint will be loaded.  检查点将从中加载的路径。
    """
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return data  # Return the loaded checkpoint data  返回加载的检查点数据


# ========================================
# Data Generators
# 数据加载
# ========================================

def train_batch_generator(
        loader: data_loader,
        batch_size: int,
        shuffle: bool = True,
) -> Iterable[Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]:
    """
    The training batch generator.  训练批次生成器。

    :param loader: The data loader instance.  数据加载器实例。
    :param batch_size: The size of each batch.  每个批次的大小。
    :param shuffle: Whether to shuffle the data before each epoch.  是否在每个 epoch 之前打乱数据。

    :return: Yields batches of (images, labels).  生成（图像，标签）批次。
    """
    # Get total number of training samples  获取训练样本的总数
    len_train = len(loader.dataset_train)

    # Create an array of indices  创建索引数组
    indices = np.arange(len_train)

    if shuffle:  # Shuffle the indices if required  如果需要，打乱索引
        np.random.shuffle(indices)

    for start_idx in range(0, len_train, batch_size):
        # Determine the end index of the batch  确定批次的结束索引
        end_idx = min(start_idx + batch_size, len_train)
        # Get the batch indices  获取批次索引
        batch_indices = indices[start_idx:end_idx]

        images_list, labels_list = [], []  # Lists to hold images and labels  用于保存图像和标签的列表

        for idx in batch_indices:
            sample = loader.__getitem__(
                index=int(idx)
            )  # Get the sample  获取样本
            if sample is None:  # Skip if sample is None  如果样本为 None，则跳过
                continue

            image, label_index, _ = sample
            images_list.append(
                image.astype(np.float32)  # Ensure image is float32  确保图像为 float32
            )
            labels_list.append(label_index)  # Append label index  添加标签索引

        if len(images_list) == 0:  # Skip if no images were loaded  如果没有加载图像，则跳过
            continue

        batch_images = jax.numpy.array(
            np.stack(
                images_list,  # Stack images into a batch  将图像堆叠成一个批次
                axis=0  # New batch dimension  新的批次维度
            )  # shape: (batch_size, H, W, C)
        )  # Convert to jax.numpy array  转换为 jax.numpy 数组
        batch_labels = jax.numpy.array(
            np.array(
                labels_list,  # Convert labels to numpy array  将标签转换为 numpy 数组
                dtype=np.int32  # Ensure labels are int32  确保标签为 int32
            )  # Convert labels to numpy array  将标签转换为 numpy 数组
        )  # Convert to jax.numpy array  转换为 jax.numpy 数组

        yield batch_images, batch_labels  # Yield the batch  生成批次  # Yield: used in generators to produce a series of values  在生成器中用于生成一系列值


def val_batch_generator(
        loader: data_loader,
        batch_size: int,
) -> Iterable[Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]:
    """
    The validation batch generator.  验证批次生成器。

    :param loader: The data loader instance.  数据加载器实例。
    :param batch_size: The size of each batch.  每个批次的大小。

    :return: Yields batches of (images, labels).  生成（图像，标签）批次。
    """
    test_items = loader.dataset_test  # List of test items  测试项列表
    # image_size = loader.get_image_size()  # Get image size  获取图像大小
    num_test = len(test_items)  # Total number of test samples  测试样本总数

    # # Function to load and preprocess a single image  加载和预处理单个图像的函数
    # def load_single_image(
    #         item: dict
    # ) -> Optional[Tuple[np.ndarray, int]]:
    #     """
    #     Loads and preprocesses a single image.  加载和预处理单个图像。
    #
    #     :param item: The test item dictionary containing 'path' and 'label_index'.  包含 'path' 和 'label_index' 的测试项字典。
    #
    #     :return: A tuple of (image_array, label_index) or None if loading fails.  （图像数组，标签索引）的元组，如果加载失败则为 None。
    #     """
    #     path = item["path"]  # Image file path  图像文件路径
    #     try:
    #         with Image.open(path) as img:
    #             img = img.convert("RGB")  # Ensure image is in RGB format  确保图像为 RGB 格式
    #             img = img.resize(image_size, Image.Resampling.LANCZOS)  # Resize image  调整图像大小
    #             arr = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]  归一化到 [0, 1]
    #         label_index = item["label_index"]  # Get label index  获取标签索引
    #         return arr, label_index  # Return image array and label index  返回图像数组和标签索引
    #     except Exception as e:
    #         print(f"[Error] Failed to load {path}: {e}")
    #         return None

    # Create an array of indices  创建索引数组
    indices = np.arange(num_test)

    # No shuffling for validation  验证不进行打乱
    for start_idx in range(0, num_test, batch_size):
        # Determine the end index of the batch  确定批次的结束索引
        end_idx = min(start_idx + batch_size, num_test)
        # Get the batch indices  获取批次索引
        batch_indices = indices[start_idx:end_idx]

        images_list, labels_list = [], []  # Lists to hold images and labels  用于保存图像和标签的列表

        for i in batch_indices:
            # result = load_single_image(test_items[int(i)])
            result = loader.__getitem__(
                index=int(i)
            )  # Get the sample  获取样本
            if result is None:
                continue
            # img, label_idx = result
            img, label_idx, _ = result
            images_list.append(img)
            labels_list.append(label_idx)

        if len(images_list) == 0:
            continue

        batch_images = jax.numpy.array(
            np.stack(
                images_list,
                axis=0
            )
        )
        batch_labels = jax.numpy.array(
            np.array(
                labels_list,
                dtype=np.int32
            )
        )

        yield batch_images, batch_labels


# ========================================
# Model Creation
# 模型创建
# ========================================

def create_model(
        model_name: str,
        num_classes: int,
        dropout_rate: Optional[float] = None,
        mamba_config: Optional[dict] = None,
) -> flax.linen.Module:
    """
    Creates and returns the specified model.  创建并返回指定的模型。

    :param model_name: The name of the model to create.  要创建的模型名称。
    :param num_classes: The number of output classes for the model.  模型的输出类别数。
    :param dropout_rate: The dropout rate for the model (if applicable).  模型的 dropout 率（如果适用）。
    :param mamba_config: Configuration dictionary for VisionMamba model (if applicable).  VisionMamba 模型的配置字典（如果适用）。

    :return: The created model instance.  创建的模型实例。
    """
    # The settings of mamba_config  mamba_config 的设置
    if model_name == "mamba" and mamba_config is None:
        mamba_config = {
            'patch_size': 16,
            'embed_dim': 128,
            'use_class_token': True,
            'depth': 4,
            'conv_kernel_size': 3,
            'ssm_expend': 2,
            'ssm_d_state': 8,
            'ssm_dt_rank': 8,
        }
    elif model_name == "vision_mamba" and mamba_config is None:
        mamba_config = {
            'patch_size': 16,
            'num_layers': 4,
            'd_model': 128,
            'd_state': 32,
        }
    elif model_name == "hybrid_mamba_cnn" and mamba_config is None:
        mamba_config = dict(
            mamba_config={
                'patch_size': 16,
                'embed_dim': 128,
                'use_class_token': True,
                'depth': 4,
                'conv_kernel_size': 3,
                'ssm_expend': 2,
                'ssm_d_state': 8,
                'ssm_dt_rank': 8,
            },
            fusion="concat_head",
            fusion_hidden=256,
        )
    elif model_name == "hybrid_mamba_resnet" and mamba_config is not None:
        mamba_config = dict(
            resnet_type="resnet18",
            mamba_config={
                'patch_size': 16,
                'embed_dim': 128,
                'use_class_token': True,
                'depth': 4,
                'conv_kernel_size': 3,
                'ssm_expend': 2,
                'ssm_d_state': 8,
                'ssm_dt_rank': 8,
            },
            fusion="concat_head",
            fusion_hidden=256,
        )

    # CNN Model  CNN 模型
    if model_name == "cnn":
        return SimpleCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    # Baseline CNN Model  基线 CNN 模型
    elif model_name == "baseline_cnn":
        return BaselineCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    # ResNet18 Models  ResNet18 模型
    elif model_name == "resnet18":
        return ResNet18(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    # ResNet34 Models  ResNet34 模型
    elif model_name == "resnet34":
        return ResNet34(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    # VisionMamba Model  VisionMamba 模型
    elif model_name == "mamba":
        return Mamba(
            num_classes=num_classes,
            patch_size=mamba_config["patch_size"],
            embed_dim=mamba_config["embed_dim"],
            use_class_token=mamba_config["use_class_token"],
            depth=mamba_config["depth"],
            conv_kernel_size=mamba_config["conv_kernel_size"],
            ssm_expend=mamba_config["ssm_expend"],
            ssm_d_state=mamba_config["ssm_d_state"],
            ssm_dt_rank=mamba_config["ssm_dt_rank"],
            dropout_rate=dropout_rate,
        )
    elif model_name == "vision_mamba":
        return VisionMamba(
            num_classes=num_classes,
            patch_size=mamba_config["patch_size"],
            num_layers=mamba_config["num_layers"],
            d_model=mamba_config["d_model"],
            dropout_rate=dropout_rate,
        )
    # Hybrid Mamba + CNN Model  混合 Mamba + CNN 模型
    elif model_name == "hybrid_mamba_cnn":
        return HybridMambaCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            cnn_dropout_rate=dropout_rate,
            mamba_config=mamba_config["mamba_config"],
            fusion=mamba_config["fusion"],
            fusion_hidden=mamba_config["fusion_hidden"],
        )
    # Hybrid Mamba + ResNet Model  混合 Mamba + ResNet 模型
    elif model_name == "hybrid_mamba_resnet":
        return HybridMambaResNet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            resnet_type=mamba_config["resnet_type"],
            mamba_config=mamba_config["mamba_config"],
            fusion=mamba_config["fusion"],
            fusion_hidden=mamba_config["fusion_hidden"],
        )
    # Unknown Model  未知模型
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ========================================
# Train State (BatchNorm support)
# 训练状态（支持 BatchNorm）
# ========================================

class TrainState(
    train_state.TrainState
):
    """
    Extended training state, including batch_stats  扩展的训练状态，包含 batch_stats

    :param batch_stats: The batch statistics for BatchNorm layers.  BatchNorm 层的批量统计信息。
    """
    batch_stats: Any


def create_train_state(
        rng: jax.random.PRNGKey,
        model: flax.linen.Module,
        image_size: tuple[int, int],
        learning_rate: float,
) -> TrainState:
    """
    Initialize training state (BatchNorm supported)  初始化训练状态（支持 BatchNorm）

    :param rng: The random number generator key.  随机数生成器密钥。
    :param model: The model instance.  模型实例。
    :param image_size: The input image size (height, width).  输入图像大小（高度，宽度）。
    :param learning_rate: The learning rate for the optimizer.  优化器的学习率。

    :return: The initialized training state.  初始化的训练状态。
    """
    # Get dummy input for initialization  获取用于初始化的虚拟输入
    H, W = image_size
    # Create a dummy batch with shape (1, H, W, 3)  创建形状为 (1, H, W, 3) 的虚拟批次
    dummy_batch = jax.numpy.zeros(
        (1, H, W, 3),
        dtype=jax.numpy.float32
    )

    # Initialization parameters 初始化参数
    variables = model.init(rng, dummy_batch, train=True)

    # Separate params and batch_stats  分离 params 和 batch_stats
    params = variables.get(
        'params',
        variables
    )
    batch_stats = variables.get(
        'batch_stats',
        {}
    )

    # Create an optimizer
    # Using Adam optimizer  使用 Adam 优化器
    # tx = optax.adam(
    #     learning_rate=learning_rate
    # )
    # Using AdamW optimizer  使用 AdamW 优化器
    tx = optax.adamw(
        learning_rate=learning_rate,
    )

    # Create training state  创建训练状态
    state = TrainState.create(
        apply_fn=model.apply,  # Model's apply function  模型的应用函数
        params=params,  # Model parameters  模型参数
        tx=tx,  # Optimizer  优化器
        batch_stats=batch_stats,  # BatchNorm statistics  BatchNorm 统计信息
    )

    return state  # Return the training state  返回训练状态


# ========================================
# Train and Eval Functions (BatchNorm support)
# 训练/验证函数（支持 BatchNorm）
# ========================================

def create_train_and_eval_functions(
        model: flax.linen.Module,
) -> Tuple[Any, Any]:
    """
    Create training and validation functions  创建训练和验证函数

    :param model: The model instance.  模型实例。

    :return: A tuple of (train_step_fn, eval_step_fn).  （train_step_fn，eval_step_fn）的元组。
    """

    def forward_pass(
            params, batch_stats,
            batch_images,
            train: bool,
            mutable: bool = False
    ) -> Any:
        """
        Performs a forward pass through the model.  通过模型执行前向传递。

        :param params: The model parameters.  模型参数。
        :param batch_stats: The batch statistics for BatchNorm layers.  BatchNorm 层的批量统计信息。
        :param batch_images: The input batch of images.  输入图像批次。
        :param train: Whether the model is in training mode.  模型是否处于训练模式。
        :param mutable: Whether to return mutable state (e.g., batch_stats).  是否返回可变状态（例如，batch_stats）。

        :return: The model outputs (and updated batch_stats if mutable).  模型输出（如果可变，则更新 batch_stats）。
        """
        # Prepare input data  准备输入数据
        input_data = batch_images
        # Create variables dictionary  创建变量字典
        variables = {'params': params, 'batch_stats': batch_stats}
        # Forward pass  前向传递
        # Mutable case:
        if mutable:
            outputs = model.apply(
                variables,  # Model variables  模型变量
                input_data,  # Input data  输入数据
                train=train,  # Training mode  训练模式
                mutable=['batch_stats'],  # Specifies which collections should be treated as mutable  指定哪些集合应视为可变
                rngs={
                    'dropout': jax.random.PRNGKey(0)  # Random key for dropout (if used)  dropout 的随机密钥（如果使用）
                }
            )
            return outputs
        # Non-mutable case  非可变情况
        else:
            logits = model.apply(
                variables,
                input_data,
                train=train,
                mutable=False,  # No mutable collections  没有可变集合
                rngs={
                    'dropout': jax.random.PRNGKey(0)
                }
            )
            return logits

    def loss_fn(
            params: jax.numpy.ndarray,
            batch_stats: Any,
            batch_images: jax.numpy.ndarray,
            batch_labels: jax.numpy.ndarray,
            weight_decay: float = 1e-4,
    ) -> Tuple[jax.numpy.ndarray, Any]:
        """
        Computes the ce_loss and updated batch statistics.  计算损失和更新的批量统计信息。

        :param params: The model parameters.  模型参数。
        :param batch_stats: The batch statistics for BatchNorm layers.  BatchNorm 层的批量统计信息。
        :param batch_images: The input batch of images.  输入图像批次。
        :param batch_labels: The true labels for the batch.  批次的真实标签。
        :param weight_decay: The weight decay factor for L2 regularization.  L2 正则化的权重衰减因子。

        :return: A tuple of (ce_loss, updated batch_stats).  （损失，更新的 batch_stats）的元组。
        """
        # Forward pass with mutable batch_stats  使用可变的 batch_stats 进行前向传递
        logits, updates = forward_pass(
            params, batch_stats, batch_images, train=True, mutable=True
        )
        # Compute loss using softmax cross-entropy  使用 softmax 交叉熵计算损失
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()
        return ce_loss, updates
        # # L2 regularization (weight decay)  L2 正则化（权重衰减）
        # l2_reg = weight_decay * sum([
        #     jax.numpy.sum(
        #         jax.numpy.square(p)
        #     ) for p in jax.tree_util.tree_leaves(params)
        # ])
        # return ce_loss + l2_reg, updates

    def clip_gradients(
            gradients: Any,
            max_norm: float = 1.0
    ) -> Any:
        """
        Clips gradients to have a maximum norm of max_norm.  将梯度裁剪为最大范数为 max_norm。

        :param gradients: The gradients to be clipped.  要裁剪的梯度。
        :param max_norm: The maximum norm for the gradients.  梯度的最大范数。

        :return: The clipped gradients.  裁剪后的梯度。
        """
        # Compute the total norm of the gradients  计算梯度的总范数
        norms = jax.tree_util.tree_map(
            lambda x: jax.numpy.linalg.norm(x),
            gradients
        )
        # Sum the norms to get the total norm  求和范数以获得总范数
        total_norm = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            norms
        )
        # Compute the clipping scale  计算裁剪比例
        scale = jax.numpy.minimum(1.0, max_norm / (total_norm + 1e-6))
        # Apply the clipping scale to the gradients  将裁剪比例应用于梯度
        clipped_gradients = jax.tree_util.tree_map(
            lambda x: x * scale,
            gradients
        )
        return clipped_gradients  # Return the clipped gradients  返回裁剪后的梯度

    @jax.jit
    def train_step(
            state: TrainState,
            batch_images, batch_labels
    ) -> Tuple[TrainState, jax.numpy.ndarray]:
        """
        Performs a single training step.  执行单个训练步骤。

        :param state: The current training state.  当前训练状态。
        :param batch_images: The input batch of images.  输入图像批次.
        :param batch_labels: The true labels for the batch.  批次的真实标签。

        :return: A tuple of (updated_state, ce_loss).  （更新后的状态，损失）的元组。
        """
        # Compute ce_loss and gradients  计算损失和梯度
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(
            state.params, state.batch_stats, batch_images, batch_labels
        )

        # Gradient clipping implementation  梯度裁剪实现
        # grads = jax.tree_util.tree_map(
        #     lambda g: jax.numpy.clip(
        #         g, -1.0, 1.0  # Clip gradients to the range [-1, 1]
        #     ),
        #     grads
        # )  # Gradient clipping  梯度裁剪
        grads = clip_gradients(grads, max_norm=1.0)

        # Update state with new parameters and batch_stats  使用新参数和 batch_stats 更新状态
        state = state.apply_gradients(
            grads=grads,
            batch_stats=updates['batch_stats']
        )

        return state, loss  # Return updated state and ce_loss  返回更新后的状态和损失

    @jax.jit
    def eval_step(
            params: jax.numpy.ndarray,
            batch_stats: Any,
            batch_images: jax.numpy.ndarray,
            batch_labels: jax.numpy.ndarray,
    ) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
        """
        Performs a single evaluation step.  执行单个评估步骤。

        :param params: The model parameters.  模型参数。
        :param batch_stats: The batch statistics for BatchNorm layers.  BatchNorm 层的批量统计信息。
        :param batch_images: The input batch of images.  输入图像批次.
        :param batch_labels: The true labels for the batch.  批次的真实标签。

        :return: A tuple of (ce_loss, accuracy).  （损失，准确率）的元组。
        """
        # Forward pass  前向传递
        logits = forward_pass(
            params, batch_stats, batch_images, train=False, mutable=False
        )

        # Compute ce_loss  计算损失
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()

        # Compute predictions  计算预测值
        preds = jax.numpy.argmax(logits, axis=-1)

        # Compute accuracy  计算准确率
        accuracy = jax.numpy.mean((preds == batch_labels).astype(jax.numpy.float32))

        return loss, accuracy  # Return ce_loss and accuracy  返回损失和准确率

    return train_step, eval_step  # Return the training and evaluation functions  返回训练和评估函数


# ========================================
# Training Loop
# 训练循环
# ========================================

def train_one_epoch(
        state: TrainState,
        loader: data_loader,
        batch_size: int,
        train_step_fn: Any,
) -> Tuple[TrainState, float, int]:
    """
    Trains the model for one epoch.  训练模型一个 epoch。

    :param state: The current training state.  当前训练状态.
    :param loader: The data loader instance.  数据加载器实例.
    :param batch_size: The size of each batch.  每个批次的大小.
    :param train_step_fn: The training step function.  训练步骤函数.

    :return: A tuple of (updated_state, average_loss, num_batches).  （更新后的状态，平均损失，批次数）的元组.
    """
    # Initialize metrics  初始化指标
    losses = []
    num_batches = 0

    # Training loop over batches  批次训练循环
    for batch_images, batch_labels in train_batch_generator(loader, batch_size, shuffle=True):
        state, loss = train_step_fn(
            state, batch_images, batch_labels
        )  # Perform a training step  执行训练步骤
        losses.append(float(loss))
        num_batches += 1

    # Compute average loss  计算平均损失
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return state, avg_loss, num_batches  # Return updated state, average loss, and number of batches  返回更新后的状态，平均损失和批次数


def validate(
        state: TrainState,
        loader: data_loader,
        batch_size: int,
        eval_step_fn: Any,
) -> Tuple[float, float, int]:
    """
    Validates the model on the validation dataset.  在验证数据集上验证模型。

    :param state: The current training state.  当前训练状态.
    :param loader: The data loader instance.  数据加载器实例.
    :param batch_size: The size of each batch.  每个批次的大小.
    :param eval_step_fn: The evaluation step function.  评估步骤函数.

    :return: A tuple of (average_loss, average_accuracy, num_samples).  （平均损失，平均准确率，样本数）的元组.
    """
    # Initialize metrics  初始化指标
    losses = []
    accuracies = []
    num_samples = 0

    # Validation loop over batches  批次验证循环
    for batch_images, batch_labels in val_batch_generator(loader, batch_size):
        loss, acc = eval_step_fn(
            state.params, state.batch_stats, batch_images, batch_labels
        )  # Perform an evaluation step  执行评估步骤
        batch_size_actual = batch_images.shape[0]  # Actual batch size (may be smaller for last batch)  实际批次大小（最后一个批次可能较小）

        losses.append(float(loss))
        accuracies.append(float(acc))
        num_samples += batch_size_actual

    # Compute average loss and accuracy  计算平均损失和准确率
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0

    return avg_loss, avg_acc, num_samples  # Return average loss, accuracy, and number of samples  返回平均损失，准确率和样本数


# ========================================
# Main Function
# 主函数
# ========================================

if __name__ == "__main__":
    # Parse command-line arguments  解析命令行参数
    parser = argparse.ArgumentParser()  # Argument parser  参数解析器
    parser.add_argument("--model", type=str, default="mamba",
                        choices=[
                            "cnn", 'resnet18', "resnet34", "mamba",
                            "baseline_cnn", "vision_mamba",
                            "hybrid_mamba_cnn", 'hybrid_mamba_resnet'
                        ],
                        help="Model architecture to use")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join("..", "data", "dataset_split", "train"),
                        help="Path to the dataset")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (height width)")
    parser.add_argument("--train_split_ratio", type=float, default=0.9,
                        help="Train/validation split ratio")
    parser.add_argument("--use_augmentation", type=bool, default=True,
                        help="Whether to use data augmentation during training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and validation")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Dropout rate for the model")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join("..", "checkpoints"),
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed for reproducibility")
    # Mamba.VisionMamba specific arguments  Mamba.VisionMamba 特定参数
    parser.add_argument("--mamba_patch_size", type=int, default=32,
                        help="Patch size for VisionMamba")
    parser.add_argument("--mamba_embed_dim", type=int, default=128,
                        help="Embedding dimension for VisionMamba")
    parser.add_argument("--mamba_use_class_token", type=bool, default=True,
                        help="Whether to use class token in VisionMamba")
    parser.add_argument("--mamba_depth", type=int, default=4,
                        help="Number of Mamba blocks in VisionMamba")
    parser.add_argument("--mamba_conv_kernel_size", type=int, default=3,
                        help="Convolution kernel size in VisionMamba")
    parser.add_argument("--mamba_ssm_expend", type=int, default=2,
                        help="SSM expend factor in VisionMamba")
    parser.add_argument("--mamba_ssm_d_state", type=int, default=8,
                        help="SSM d_state in VisionMamba")
    parser.add_argument("--mamba_ssm_dt_rank", type=int, default=8,
                        help="SSM dt_rank in VisionMamba")
    # VisionMamba.VisionMamba specific arguments  VisionMamba.VisionMamba 特定参数
    parser.add_argument("--vision_mamba_patch_size", type=int, default=16,
                        help="Patch size for VisionMamba")
    parser.add_argument("--vision_mamba_num_layers", type=int, default=4,
                        help="Number of layers in VisionMamba")
    parser.add_argument("--vision_mamba_d_model", type=int, default=128,
                        help="Model dimension in VisionMamba")
    parser.add_argument("--vision_mamba_d_state", type=int, default=32,
                        help="SSM d_state in VisionMamba")
    # Hybrid Mamba + CNN/ResNet specific arguments  混合 Mamba + CNN 特定参数
    parser.add_argument("--hybrid_fusion", type=str, default="weighted_sum",
                        choices=["concat_head", "weighted_sum", "gated_sum"],
                        help="Fusion method for Hybrid Mamba + CNN model")
    parser.add_argument("--hybrid_fusion_hidden", type=int, default=256,
                        help="Hidden dimension for fusion in Hybrid Mamba + CNN/ResNet model")
    parser.add_argument("--hybrid_resnet_type", type=str, default="resnet18",
                        choices=["resnet18", "resnet34"],
                        help="ResNet type for Hybrid Mamba + ResNet model")
    # Parse the arguments  解析参数
    args = parser.parse_args()

    # Set random seeds for reproducibility  设置随机种子以确保可重复性
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Prepare checkpoint directory  准备检查点目录
    image_size = tuple(args.image_size)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    model_ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    os.makedirs(model_ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model         : {args.model}")
    print(f"Data path     : {os.path.abspath(args.data_path)}")
    print(f"Image size    : {image_size}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.num_epochs}")
    print(f"Learning rate :  {args.learning_rate}")
    print(f"Checkpoint dir: {os.path.abspath(model_ckpt_dir)}")
    print("=" * 60)

    # Loading data 加载数据
    print("\n[1/4] Loading dataset...")
    loader = data_loader(
        data_path=args.data_path,
        image_size=image_size,
        train_split_ratio=args.train_split_ratio,
        use_augmentation=args.use_augmentation,
    )  # Data loader instance  数据加载器实例

    num_classes = loader.get_num_classes()
    unique_labels = loader.get_unique_labels()
    train_size = len(loader.dataset_train)
    val_size = len(loader.dataset_test)

    print(f"  Classes  : {num_classes}")
    print(f"  Labels   : {unique_labels}")
    print(f"  Train/Val: {train_size} / {val_size}")

    # Create the model  创建模型
    print("\n[2/4] Creating model...")
    mamba_config = None
    if args.model == "mamba":
        mamba_config = {
            'patch_size': args.mamba_patch_size,
            'embed_dim': args.mamba_embed_dim,
            'use_class_token': args.mamba_use_class_token,
            'depth': args.mamba_depth,
            'conv_kernel_size': args.mamba_conv_kernel_size,
            'ssm_expend': args.mamba_ssm_expend,
            'ssm_d_state': args.mamba_ssm_d_state,
            'ssm_dt_rank': args.mamba_ssm_dt_rank,
        }
    elif args.model == "vision_mamba":
        mamba_config = {
            'patch_size': args.vision_mamba_patch_size,
            'num_layers': args.vision_mamba_num_layers,
            'd_model': args.vision_mamba_d_model,
            'd_state': args.vision_mamba_d_state,
        }
    elif args.model == "hybrid_mamba_cnn":
        mamba_config = dict(
            mamba_config={
                'patch_size': args.mamba_patch_size,
                'embed_dim': args.mamba_embed_dim,
                'use_class_token': args.mamba_use_class_token,
                'depth': args.mamba_depth,
                'conv_kernel_size': args.mamba_conv_kernel_size,
                'ssm_expend': args.mamba_ssm_expend,
                'ssm_d_state': args.mamba_ssm_d_state,
                'ssm_dt_rank': args.mamba_ssm_dt_rank,
            },
            fusion=args.hybrid_fusion,
            fusion_hidden=args.hybrid_fusion_hidden,
        )
    elif args.model == "hybrid_mamba_resnet":
        mamba_config = dict(
            resnet_type=args.hybrid_resnet_type,
            mamba_config={
                'patch_size': args.mamba_patch_size,
                'embed_dim': args.mamba_embed_dim,
                'use_class_token': args.mamba_use_class_token,
                'depth': args.mamba_depth,
                'conv_kernel_size': args.mamba_conv_kernel_size,
                'ssm_expend': args.mamba_ssm_expend,
                'ssm_d_state': args.mamba_ssm_d_state,
                'ssm_dt_rank': args.mamba_ssm_dt_rank,
            },
            fusion=args.hybrid_fusion,
            fusion_hidden=args.hybrid_fusion_hidden,
        )
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        mamba_config=mamba_config,  # VisionMamba 配置
    )

    # Initialize training state  初始化训练状态
    print("\n[3/4] Initializing train state...")
    state = create_train_state(rng, model, image_size, args.learning_rate)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Total params: {num_params:,}")

    # Create training/validation functions  创建训练/验证函数
    train_step_fn, eval_step_fn = create_train_and_eval_functions(model)

    # Training loop  训练循环
    print("\n[4/4] Start training...")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0

    train_loss_list, val_loss_list, val_acc_list = [], [], []

    for epoch in tqdm(range(1, args.num_epochs + 1)):
        epoch_start = time.time()

        # Train 训练
        loader.set_last(
            is_last=False if epoch < args.num_epochs else True  # 最后一个 epoch 设置为最后一次迭代
        )  # Indicate if it's the last epoch  指示是否为最后一个 epoch
        loader.set_training(
            is_training=True,  # 训练模式
        )  # Set loader to training mode  将加载器设置为训练模式
        state, train_loss, n_train_batches = train_one_epoch(
            state, loader, args.batch_size, train_step_fn
        )  # Train for one epoch  训练一个 epoch

        # Validation 验证
        loader.set_training(
            is_training=False,  # 验证模式
        )  # Set loader to validation mode  将加载器设置为验证模式
        val_loss, val_acc, n_val_samples = validate(
            state, loader, args.batch_size, eval_step_fn
        )  # Validate the model  验证模型

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} ({n_train_batches} batches)")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f} ({n_val_samples} samples)")

        # Save the best model  保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            ckpt = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'epoch': epoch,
                'best_acc': best_val_acc,
            }

            best_ckpt_path = os.path.join(model_ckpt_dir, "best.pkl")
            save_checkpoint(best_ckpt_path, ckpt)
            print(f"  ✓ Best model saved (acc={best_val_acc:.4f})")

        # Regularly save  定期保存
        if args.save_every != 0 and epoch % args.save_every == 0:
            ckpt = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'epoch': epoch,
            }

            epoch_ckpt_path = os.path.join(model_ckpt_dir, f"epoch_{epoch}.pkl")
            save_checkpoint(epoch_ckpt_path, ckpt)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # Draw training curves  绘制训练曲线
        if epoch >= 2 or epoch == args.num_epochs:  # Draw after 2 epochs or last epoch  绘制在 2 个 epoch 后或最后一个 epoch
            plt.figure(figsize=(12, 4))
            epoch_xlist = range(1, epoch + 1)
            # Loss subplot  损失子图
            plt.subplot(1, 2, 1)
            plt.plot(epoch_xlist, train_loss_list, label='Train Loss')
            plt.plot(epoch_xlist, val_loss_list, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            # Accuracy subplot 准确率子图
            plt.subplot(1, 2, 2)
            plt.plot(epoch_xlist, val_acc_list, label='Val Accuracy', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_ckpt_dir, 'training_curves.png'))
            plt.close()

    print("\n" + "=" * 60)
    print(f"Training done! Best acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("=" * 60)
