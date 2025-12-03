"""
伤口分类测试脚本 - 仅评估，不训练
使用 JAX + Flax + Optax
支持模型：CNN / ResNet34 / Mamba
从指定 checkpoint 读取模型参数，对图片 dataset 进行测试评估
"""

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

# 设置路径，保证可以 import scripts 和 nets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from scripts.dataset import data_loader
from nets.CNN import SimpleCNN
from nets.ResNet import ResNet18, ResNet34
from nets.Mamba import VisionMamba

import pickle


# ========================================
# Checkpoint 工具（只需要 load）
# ========================================

def load_checkpoint(ckpt_path: str) -> dict:
    """加载 checkpoint（pickle 格式）"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return data


# ========================================
# 验证集批次生成器（和 train.py 一致）
# ========================================

def val_batch_generator(
        loader: data_loader,
        batch_size: int,
) -> Iterable[Tuple[jax.numpy.ndarray, jax.numpy.ndarray]]:
    """验证/测试集批次生成器"""
    test_items = loader.dataset_test
    image_size = loader.get_image_size()
    num_test = len(test_items)

    def load_single_image(item):
        path = item["path"]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                arr = np.array(img).astype(np.float32) / 255.0
            label_index = item["label_index"]
            return arr, label_index
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            return None

    indices = np.arange(num_test)

    for start_idx in range(0, num_test, batch_size):
        end_idx = min(start_idx + batch_size, num_test)
        batch_indices = indices[start_idx:end_idx]

        images_list = []
        labels_list = []

        for i in batch_indices:
            result = load_single_image(test_items[int(i)])
            if result is None:
                continue
            img, label_idx = result
            images_list.append(img)
            labels_list.append(label_idx)

        if len(images_list) == 0:
            continue

        batch_images = jax.numpy.array(np.stack(images_list, axis=0))
        batch_labels = jax.numpy.array(np.array(labels_list, dtype=np.int32))

        yield batch_images, batch_labels


# ========================================
# 模型创建（和 train.py 一致）
# ========================================

def create_model(
        model_name: str,
        num_classes: int,
        image_size: Tuple[int, int],
        mamba_config: Optional[dict] = None,
):
    """根据名称创建模型"""
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)

    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes)

    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes)

    elif model_name == "mamba":
        if mamba_config is None:
            mamba_config = {
                'patch_size': 16,
                'embed_dim': 512,
                'use_class_token': False,
                'depth': 8,
                'conv_kernel_size': 3,
                'ssm_expend': 2,
                'ssm_d_state': 16,
                'ssm_dt_rank': 16,
            }

        return VisionMamba(
            num_classes=num_classes,
            patch_size=mamba_config["patch_size"],
            embed_dim=mamba_config["embed_dim"],
            use_class_token=mamba_config["use_class_token"],
            depth=mamba_config["depth"],
            conv_kernel_size=mamba_config["conv_kernel_size"],
            ssm_expend=mamba_config["ssm_expend"],
            ssm_d_state=mamba_config["ssm_d_state"],
            ssm_dt_rank=mamba_config["ssm_dt_rank"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ========================================
# 主测试流程
# ========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mamba",
                        choices=["cnn", "resnet18", "resnet34", "mamba"],
                        help="Model architecture to use")  # 使用的模型架构
    parser.add_argument("--data_path", type=str,
                        default=os.path.join("..", "data", "dataset"),
                        help="Path to the dataset")  # 数据集路径
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (height width)")  # 输入图像大小（高 宽）
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")  # 评估时的批次大小
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Specify the checkpoint path; if empty, use ../checkpoints/<model>/best.pkl")  # 指定 checkpoint 路径，如果为空则使用 ../checkpoints/<model>/best.pkl
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed for reproducibility")  # 用于可重复性的随机种子

    # Mamba related parameters (must remain consistent with those used during training) Mamba 相关参数（需要与训练时保持一致）
    parser.add_argument("--mamba_patch_size", type=int, default=16,
                        help="Patch size used in Mamba model")  # Mamba 模型中使用的 Patch 大小
    parser.add_argument("--mamba_embed_dim", type=int, default=256,
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

    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

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

    # -------------------------
    # [1/3] 加载数据
    # -------------------------
    print("\n[1/3] Loading dataset...")
    loader = data_loader(
        data_path=args.data_path,
        image_size=image_size,
        train_split_ratio=0.0,  # Fixed division ratio during testing, all for testing  # 测试时固定划分比例，全用于测试
        use_augmentation=False,  # No augmentation during testing  # 测试时不进行增强
    )

    num_classes = loader.get_num_classes()
    unique_labels = loader.get_unique_labels()
    test_size = len(loader.dataset_test)

    print(f"  Classes : {num_classes}")
    print(f"  Labels  : {unique_labels}")
    print(f"  Test size: {test_size} images")

    # -------------------------
    # [2/3] 创建模型 & 加载 checkpoint
    # -------------------------
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

    model = create_model(args.model, num_classes, image_size, mamba_config)

    # 这里只需要加载参数，不需要重新初始化 TrainState
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt.get("params", None)
    batch_stats = ckpt.get("batch_stats", {})

    if params is None:
        raise ValueError("Checkpoint 中未找到 'params' 字段，请检查训练保存逻辑。")

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Loaded params: {num_params:,}")
    if "epoch" in ckpt:
        print(f"  Trained epoch: {ckpt['epoch']}")
    if "best_acc" in ckpt:
        print(f"  Best val acc : {ckpt['best_acc']:.4f}")


    # 定义单步评估（JIT 加速）
    @jax.jit
    def eval_step(p, b_stats, batch_images, batch_labels):
        variables = {"params": p, "batch_stats": b_stats}
        logits = model.apply(
            variables,
            batch_images,
            train=False,
            mutable=False
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((preds == batch_labels).astype(jnp.float32))
        return loss, acc, preds


    # -------------------------
    # [3/3] 在测试集上评估
    # -------------------------
    print("\n[3/3] Evaluating on test/validation set...")
    start_time = time.time()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 混淆矩阵 & 每类统计
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    num_batches = max(1, (test_size + args.batch_size - 1) // args.batch_size)

    for batch_images, batch_labels in tqdm(
            val_batch_generator(loader, args.batch_size),
            total=num_batches,
            desc="Evaluating",
    ):
        loss, acc, preds = eval_step(params, batch_stats, batch_images, batch_labels)

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

    # 每类准确率
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

    # 打印（小号）混淆矩阵
    print("\nConfusion matrix (rows = true, cols = pred):")
    np.set_printoptions(linewidth=120)
    print(confusion)

    print("\n" + "=" * 60)
    print("Evaluation done.")
    print("=" * 60)
