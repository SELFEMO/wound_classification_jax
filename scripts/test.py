import os
import sys
import argparse
import time
from typing import Tuple, Optional, Iterable

import numpy as np
import jax
import jax.numpy as jnp
import optax
from PIL import Image
from tqdm import tqdm

# Set the path to ensure that scripts and nets can be imported.  设置路径，保证可以 import scripts 和 nets。
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from scripts.dataset import data_loader

from nets.CNN import SimpleCNN
from nets.ResNet import ResNet18, ResNet34
# from nets.Mamba import VisionMamba
from nets.Mamba import VisionMamba as Mamba

from nets.BaselineCNN import BaselineCNN
from nets.VisionMamba import VisionMamba

import pickle


# ========================================
# Checkpoint Tools
# Checkpoint 工具
# ========================================

def load_checkpoint(
        ckpt_path: str
) -> dict:
    """
    Load checkpoint (pickle format)  加载 checkpoint（pickle 格式）

    :param ckpt_path: Path to the checkpoint file  checkpoint 文件的路径

    :return: Loaded checkpoint data  加载的 checkpoint 数据
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)  # Use pickle to load the checkpoint  使用 pickle 加载 checkpoint
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return data


# ========================================
# Validation set batch generator (same as train.py)
# 验证集批次生成器（和 train.py 一致）
# ========================================

# def val_batch_generator(
#         loader: data_loader,
#         batch_size: int,
# ) -> Iterable[Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]:
#     """
#     Validation set batch generator  验证集批次生成器
#
#     :param loader: Data loader instance  数据加载器实例
#     :param batch_size: Batch size  批次大小
#
#     :return: Yields batches of (images, labels)  生成（图像，标签）批次
#     """
#     # Get test dataset items and image size  获取测试数据集条目和图像大小
#     test_items = loader.dataset_test
#     # image_size = loader.get_image_size()
#     num_test = len(test_items)
#
#     # # 单张图像加载和预处理函数
#     # def load_single_image(
#     #         item: dict
#     # ) -> Optional[Tuple[np.ndarray, int]]:
#     #     """
#     #     Load and preprocess a single image  加载和预处理单张图像
#     #
#     #     :param item: Dictionary containing image path and label index  包含图像路径和标签索引的字典
#     #
#     #     :return: Tuple of (image array, label index) or None if loading fails  图像数组和标签索引的元组，如果加载失败则返回 None
#     #     """
#     #     path = item["path"]
#     #     try:
#     #         with Image.open(path) as img:
#     #             img = img.convert("RGB")
#     #             img = img.resize(image_size, Image.Resampling.LANCZOS)
#     #             arr = np.array(img).astype(np.float32) / 255.0
#     #         label_index = item["label_index"]
#     #         return arr, label_index
#     #     except Exception as e:
#     #         print(f"[Error] Failed to load {path}: {e}")
#     #         return None
#
#     # Generate batches  生成批次
#     indices = np.arange(num_test)
#
#     # No shuffling during validation/testing  验证/测试时不进行打乱
#     for start_idx in range(0, num_test, batch_size):
#         # Create batch  创建批次
#         end_idx = min(start_idx + batch_size, num_test)
#         batch_indices = indices[start_idx:end_idx]
#
#         images_list, labels_list = [], []
#
#         for i in batch_indices:
#             # result = load_single_image(test_items[int(i)])
#             result = loader.__getitem__(
#                 index=int(i)
#             )  # Get the sample  获取样本
#             if result is None:
#                 continue
#             # img, label_idx = result
#             img, label_idx, _ = result
#             # Append to batch lists  添加到批次列表
#             images_list.append(img)
#             labels_list.append(label_idx)
#
#         if len(images_list) == 0:  # If no images were loaded successfully, skip this batch  如果没有图像加载成功，则跳过此批次
#             continue
#
#         batch_images = jax.numpy.array(
#             np.stack(
#                 images_list,  # Stack images along a new axis  # 沿新轴堆叠图像
#                 axis=0  # Batch dimension  # 批次维度
#             )  # shape: (batch_size, height, width, channels)
#         )  # Convert to JAX array  转换为 JAX 数组
#         batch_labels = jax.numpy.array(
#             np.array(
#                 labels_list,  # Convert list to NumPy array  将列表转换为 NumPy 数组
#                 dtype=np.int32
#             )  # Convert to NumPy array first, then to JAX array  # 先转换为 NumPy 数组，然后转换为 JAX 数组
#         )  # shape: (batch_size,)
#
#         yield batch_images, batch_labels  # Yield the batch  生成批次

from scripts.train import val_batch_generator

# ========================================
# Model creation (same as train.py)
# 模型创建（和 train.py 一致）
# ========================================

# def create_model(
#         model_name: str,
#         num_classes: int,
#         mamba_config: Optional[dict] = None,
# ) -> jax.nn.Module:
#     """
#     Create model instance based on the specified architecture  根据指定的架构创建模型实例
#
#     :param model_name: Name of the model architecture  模型架构名称
#     :param num_classes: Number of output classes  输出类别数
#     :param mamba_config: Configuration dictionary for Mamba model (if applicable)  Mamba 模型的配置字典（如果适用）
#
#     :return: Model instance  模型实例
#     """
#     # CNN model  CNN 模型
#     if model_name == "cnn":
#         return SimpleCNN(num_classes=num_classes)
#     # Baseline CNN model  基线 CNN 模型
#     elif model_name == "baseline_cnn":
#         return BaselineCNN(num_classes=num_classes)
#     # ResNet18 models  ResNet18 模型
#     elif model_name == "resnet18":
#         return ResNet18(num_classes=num_classes)
#     # ResNet34 models  ResNet34 模型
#     elif model_name == "resnet34":
#         return ResNet34(num_classes=num_classes)
#     # Mamba model  Mamba 模型
#     elif model_name == "mamba":
#         if mamba_config is None:
#             mamba_config = {
#                 'patch_size': 16,
#                 'embed_dim': 512,
#                 'use_class_token': False,
#                 'depth': 8,
#                 'conv_kernel_size': 3,
#                 'ssm_expend': 2,
#                 'ssm_d_state': 16,
#                 'ssm_dt_rank': 16,
#             }
#         # return VisionMamba(
#         return Mamba(
#             num_classes=num_classes,
#             patch_size=mamba_config["patch_size"],
#             embed_dim=mamba_config["embed_dim"],
#             use_class_token=mamba_config["use_class_token"],
#             depth=mamba_config["depth"],
#             conv_kernel_size=mamba_config["conv_kernel_size"],
#             ssm_expend=mamba_config["ssm_expend"],
#             ssm_d_state=mamba_config["ssm_d_state"],
#             ssm_dt_rank=mamba_config["ssm_dt_rank"],
#         )
#     # VisionMamba model  VisionMamba 模型
#     elif model_name == "vision_mamba":
#         if mamba_config is None:
#             mamba_config = {
#                 'patch_size': 16,
#                 'num_layers': 4,
#                 'd_model': 128,
#                 'd_state': 32,
#             }
#         return VisionMamba(
#             num_classes=num_classes,
#             patch_size=mamba_config["patch_size"],
#             num_layers=mamba_config["num_layers"],
#             d_model=mamba_config["d_model"],
#         )
#     # Unknown model  未知模型
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

from scripts.train import create_model

# ========================================
# Main testing flow
# 主测试流程
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
                        help="Model architecture to use")  # 使用的模型架构
    parser.add_argument("--data_path", type=str,
                        default=os.path.join("..", "data", "dataset_split", "test"),
                        help="Path to the dataset")  # 数据集路径
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (height width)")  # 输入图像大小（高 宽）
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")  # 评估时的批次大小
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Specify the checkpoint path; if empty, use ../checkpoints/<model>/best.pkl")  # 指定 checkpoint 路径，如果为空则使用 ../checkpoints/<model>/best.pkl
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed for reproducibility")  # 用于可重复性的随机种子
    # Mamba.VisionMamba related parameters (must remain consistent with those used during training) Mamba.VisionMamba 相关参数（需要与训练时保持一致）
    parser.add_argument("--mamba_patch_size", type=int, default=32,
                        help="Patch size used in Mamba model")  # Mamba 模型中使用的 Patch 大小
    parser.add_argument("--mamba_embed_dim", type=int, default=128,
                        help="Embedding dimension used in Mamba model")  # Mamba 模型中使用的嵌入维度
    parser.add_argument("--mamba_use_class_token", type=bool, default=True,
                        help="Whether to use class token in Mamba model")  # Mamba 模型中是否使用类别标记
    parser.add_argument("--mamba_depth", type=int, default=4,
                        help="Depth (number of layers) of the Mamba model")  # Mamba 模型的深度（层数）
    parser.add_argument("--mamba_conv_kernel_size", type=int, default=3,
                        help="Convolution kernel size in Mamba model")  # Mamba 模型中的卷积核大小
    parser.add_argument("--mamba_ssm_expend", type=int, default=2,
                        help="SSM expend factor in Mamba model")  # Mamba 模型中的 SSM 扩展因子
    parser.add_argument("--mamba_ssm_d_state", type=int, default=8,
                        help="SSM d_state in Mamba model")  # Mamba 模型中的 SSM d_state
    parser.add_argument("--mamba_ssm_dt_rank", type=int, default=8,
                        help="SSM dt_rank in Mamba model")  # Mamba 模型中的 SSM dt_rank
    # VisionMamba.VisionMamba related parameters (must remain consistent with those used during training) VisionMamba.VisionMamba 相关参数（需要与训练时保持一致）
    parser.add_argument("--vision_mamba_patch_size", type=int, default=16,
                        help="Patch size used in VisionMamba model")  # VisionMamba 模型中使用的 Patch 大小
    parser.add_argument("--vision_mamba_num_layers", type=int, default=4,
                        help="Number of layers in VisionMamba model")  # VisionMamba 模型中的层数
    parser.add_argument("--vision_mamba_d_model", type=int, default=128,
                        help="Dimension of model in VisionMamba model")  # VisionMamba 模型中的模型维度
    parser.add_argument("--vision_mamba_d_state", type=int, default=32,
                        help="Dimension of state in VisionMamba model")  # VisionMamba 模型中的状态维度
    # Hybrid Mamba + CNN/ResNet specific arguments  混合 Mamba + CNN 特定参数
    parser.add_argument("--hybrid_fusion", type=str, default="concat_head",
                        choices=["concat_head", "weighted_sum", "gated_sum"],
                        help="Fusion method for Hybrid Mamba + CNN model")  # 混合 Mamba + CNN 模型的融合方法
    parser.add_argument("--hybrid_fusion_hidden", type=int, default=256,
                        help="Hidden dimension for fusion in Hybrid Mamba + CNN/ResNet model")  # 混合 Mamba + CNN/ResNet 模型中融合的隐藏维度
    parser.add_argument("--hybrid_resnet_type", type=str, default="resnet18",
                        choices=["resnet18", "resnet34"],
                        help="ResNet type for Hybrid Mamba + ResNet model")  # 混合 Mamba + ResNet 模型的 ResNet 类型
    # Parse the arguments  解析参数
    args = parser.parse_args()

    # Set random seed for reproducibility  设置随机种子以实现可重复性
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Print configuration  打印配置信息
    image_size = tuple(args.image_size)

    # 默认 checkpoint 路径：../checkpoints/<model>/best.pkl
    if args.ckpt_path is None:
        default_ckpt_dir = os.path.join("..", "checkpoints", args.model)
        ckpt_path = os.path.join(default_ckpt_dir, "best.pkl")
    else:
        ckpt_path = args.ckpt_path

    print("=" * 60)
    print(f"Model         : {args.model}")
    print(f"Data path     : {os.path.abspath(args.data_path)}")
    print(f"Image size    : {image_size}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Checkpoint    : {os.path.abspath(ckpt_path)}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # [1/3] Loading data  加载数据
    # ---------------------------------------------------------------
    print("\n[1/3] Loading dataset...")
    loader = data_loader(
        data_path=args.data_path,
        image_size=image_size,
        train_split_ratio=0.0,  # Fixed division ratio during testing, all for testing  # 测试时固定划分比例，全用于测试
        use_augmentation=False,  # No augmentation during testing  # 测试时不进行增强
    )  # Initialize data loader  初始化数据加载器
    loader.set_training(
        is_training=False,  # Testing mode  # 测试模式
    )  # Set to evaluation mode  设置为评估模式

    num_classes = loader.get_num_classes()
    unique_labels = loader.get_unique_labels()
    test_size = len(loader.dataset_test)

    print(f"  Classes : {num_classes}")
    print(f"  Labels  : {unique_labels}")
    print(f"  Test size: {test_size} images")

    # ---------------------------------------------------------------
    # [2/3] Create model & load checkpoint  创建模型 & 加载 checkpoint
    # ---------------------------------------------------------------
    print("\n[2/3] Creating model and loading checkpoint...")

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
        dropout_rate=None,
        mamba_config=mamba_config,
    )

    # Here, we only need to load the parameters; we don't need to reinitialize TrainState.  这里只需要加载参数，不需要重新初始化 TrainState
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt.get("params", None)
    batch_stats = ckpt.get("batch_stats", {})

    if params is None:
        raise ValueError("The 'params' field was not found in the checkpoint. Please check the training and saving logic.")  # Checkpoint 中未找到 'params' 字段，请检查训练保存逻辑。

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Loaded params: {num_params:,}")
    if "epoch" in ckpt:
        print(f"  Trained epoch: {ckpt['epoch']}")
    if "best_acc" in ckpt:
        print(f"  Best val acc : {ckpt['best_acc']:.4f}")


    # Define single-step evaluation (JIT acceleration)  定义单步评估（JIT 加速）
    @jax.jit
    def eval_step(
            p: dict,
            b_stats: dict,
            batch_images: jax.numpy.ndarray,
            batch_labels: jax.numpy.ndarray,
    ) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]:
        """
        Single evaluation step  单步评估

        :param p: The model parameters  模型参数
        :param b_stats: The model batch statistics  模型批次统计
        :param batch_images: The input batch images  输入批次图像
        :param batch_labels: The input batch labels  输入批次标签

        :return: Tuple of (loss, accuracy, predictions)  （损失，准确率，预测）的元组
        """
        # Combine parameters and batch statistics  结合参数和批次统计
        variables = {"params": p, "batch_stats": b_stats}
        logits = model.apply(
            variables,  # 模型变量
            batch_images,  # Input images  输入图像
            train=False,  # Evaluation mode  评估模式
            mutable=False,  # No mutable state  无可变状态
            # rngs={
            #     "dropout": jax.random.PRNGKey(0)  # Fixed dropout key during evaluation  评估期间固定 dropout 密钥
            # }
        )  # Forward pass  前向传播
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()  # Compute loss  计算损失
        preds = jnp.argmax(logits, axis=-1)  # Predicted classes  预测类别
        acc = jnp.mean((preds == batch_labels).astype(jnp.float32))  # Compute accuracy  计算准确率
        return loss, acc, preds  # Return loss, accuracy, and predictions  返回损失、准确率和预测结果


    # ---------------------------------------------------------------
    # [3/3] Evaluate on the test set  在测试集上评估
    # ---------------------------------------------------------------
    print("\n[3/3] Evaluating on test/validation set...")
    start_time = time.time()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Confusion Matrix & Class Statistics  混淆矩阵 & 每类统计
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    num_batches = max(1, (test_size + args.batch_size - 1) // args.batch_size)

    for batch_images, batch_labels in tqdm(
            val_batch_generator(loader, args.batch_size),
            total=num_batches,
            desc="Evaluating",
    ):  # Iterate over validation batches  遍历验证批次
        loss, acc, preds = eval_step(
            params,
            batch_stats,
            batch_images,
            batch_labels
        )  # Evaluate batch  评估批次

        loss_val = float(loss)
        preds_np = np.array(preds)
        labels_np = np.array(batch_labels)

        batch_size_actual = labels_np.shape[0]
        total_loss += loss_val * batch_size_actual
        total_correct += int((preds_np == labels_np).sum())
        total_samples += batch_size_actual

        # 更新混淆矩阵
        for t, p in zip(labels_np, preds_np):
            confusion[int(t), int(p)] += 1

    elapsed = time.time() - start_time

    if total_samples == 0:
        print("No samples in test set. Please check your data path and split ratio.")
        sys.exit(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    print("\n" + "=" * 60)
    print("Test / Validation Result")
    print("=" * 60)
    print(f"  Samples      : {total_samples}")
    print(f"  Avg loss     : {avg_loss:.4f}")
    print(f"  Overall acc  : {avg_acc:.4f}")
    print(f"  Time         : {elapsed:.1f} s")
    print("=" * 60)

    # Accuracy per category  每类准确率
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        label_name = unique_labels[i] if i < len(unique_labels) else f"class_{i}"
        support = confusion[i].sum()
        correct = confusion[i, i]
        if support > 0:
            acc_i = correct / support
            print(f"  [{i}] {label_name:20s} | acc = {acc_i:.4f} ({correct}/{support})")
        else:
            print(f"  [{i}] {label_name:20s} | acc = N/A   (0 samples)")

    # Print the (small) confusion matrix  打印（小号）混淆矩阵
    print("\nConfusion matrix (rows = true, cols = pred):")
    np.set_printoptions(linewidth=120)
    print(confusion)

    print("\n" + "=" * 60)
    print("Evaluation done.")
    print("=" * 60)
