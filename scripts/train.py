# train.py (无外部 checkpoint 依赖版本)
"""
伤口分类训练脚本 - 支持 BatchNorm
使用 JAX + Flax + Optax
支持模型：CNN / ResNet34 / Mamba
使用 pickle 保存 checkpoint（无需 Orbax 和 TensorFlow）
"""

import os
import sys
import argparse
from typing import Tuple, Iterable, Optional, Any
import time
import pickle
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import linen as nn
import optax
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from scripts.dataset import data_loader
from nets.CNN import SimpleCNN
from nets.ResNet import ResNet18, ResNet34
from nets.Mamba import VisionMamba


# ========================================
# 简易 Checkpoint 工具（使用 pickle）
# ========================================

def save_checkpoint(ckpt_path: str, data: dict):
    """保存 checkpoint"""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(ckpt_path: str) -> dict:
    """加载 checkpoint"""
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return data


# ========================================
# 数据加载
# ========================================

def train_batch_generator(
        loader: data_loader,
        batch_size: int,
        shuffle: bool = True,
) -> Iterable[Tuple[jnp.ndarray, jnp.ndarray]]:
    """训练集批次生成器"""
    len_train = len(loader.dataset_train)
    indices = np.arange(len_train)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len_train, batch_size):
        end_idx = min(start_idx + batch_size, len_train)
        batch_indices = indices[start_idx:end_idx]

        images_list = []
        labels_list = []

        for idx in batch_indices:
            sample = loader.__getitem__(int(idx))
            if sample is None:
                continue

            image, label_index, _ = sample
            images_list.append(image.astype(np.float32))
            labels_list.append(label_index)

        if len(images_list) == 0:
            continue

        batch_images = jnp.array(np.stack(images_list, axis=0))
        batch_labels = jnp.array(np.array(labels_list, dtype=np.int32))

        yield batch_images, batch_labels


def val_batch_generator(
        loader: data_loader,
        batch_size: int,
) -> Iterable[Tuple[jnp.ndarray, jnp.ndarray]]:
    """验证集批次生成器"""
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

        batch_images = jnp.array(np.stack(images_list, axis=0))
        batch_labels = jnp.array(np.array(labels_list, dtype=np.int32))

        yield batch_images, batch_labels


# ========================================
# 模型创建
# ========================================

def create_model(
        model_name: str,
        num_classes: int,
        image_size: Tuple[int, int],
        mamba_config: Optional[dict] = None,
) -> nn.Module:
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
                'dropout_rate': 0.1,
                'depth': 8,
                'conv_kernel_size': 3,
            }

        return VisionMamba(
            num_classes=num_classes,
            patch_size=mamba_config["patch_size"],
            embed_dim=mamba_config["embed_dim"],
            use_class_token=mamba_config["use_class_token"],
            dropout_rate=mamba_config["dropout_rate"],
            depth=mamba_config["depth"],
            conv_kernel_size=mamba_config["conv_kernel_size"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ========================================
# 训练状态（支持 BatchNorm）
# ========================================

class TrainState(train_state.TrainState):
    """扩展的训练状态，包含 batch_stats"""
    batch_stats: Any


def create_train_state(
        rng: jax.random.PRNGKey,
        model: nn.Module,
        model_name: str,
        image_size: Tuple[int, int],
        learning_rate: float,
) -> TrainState:
    """初始化训练状态（支持 BatchNorm）"""
    H, W = image_size
    dummy_batch = jnp.zeros((1, H, W, 3), dtype=jnp.float32)

    # 初始化参数
    if model_name == "mamba":
        B, H, W, C = dummy_batch.shape
        dummy_seq = dummy_batch.reshape(B, H * W, C)
        variables = model.init(rng, dummy_seq, train=True)
    else:
        variables = model.init(rng, dummy_batch, train=True)

    # 分离 params 和 batch_stats
    params = variables.get('params', variables)
    batch_stats = variables.get('batch_stats', {})

    # 创建优化器
    tx = optax.adam(learning_rate)

    # 创建训练状态
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state


# ========================================
# 训练/验证函数（支持 BatchNorm）
# ========================================

def create_train_and_eval_functions(model: nn.Module, model_name: str):
    """创建训练和验证函数"""

    def forward_pass(params, batch_stats, batch_images, train: bool, mutable: bool = False):
        if model_name == "mamba":
            B, H, W, C = batch_images.shape
            seq = batch_images.reshape(B, H * W, C)
            input_data = seq
        else:
            input_data = batch_images

        variables = {'params': params, 'batch_stats': batch_stats}

        if mutable:
            outputs = model.apply(
                variables,
                input_data,
                train=train,
                mutable=['batch_stats']
            )
            return outputs
        else:
            logits = model.apply(
                variables,
                input_data,
                train=train,
                mutable=False
            )
            return logits

    def loss_fn(params, batch_stats, batch_images, batch_labels):
        logits, updates = forward_pass(
            params, batch_stats, batch_images, train=True, mutable=True
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()
        return loss, updates

    @jax.jit
    def train_step(state: TrainState, batch_images, batch_labels):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(
            state.params, state.batch_stats, batch_images, batch_labels
        )

        state = state.apply_gradients(
            grads=grads,
            batch_stats=updates['batch_stats']
        )

        return state, loss

    @jax.jit
    def eval_step(params, batch_stats, batch_images, batch_labels):
        logits = forward_pass(
            params, batch_stats, batch_images, train=False, mutable=False
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_labels,
        ).mean()

        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean((preds == batch_labels).astype(jnp.float32))

        return loss, accuracy

    return train_step, eval_step


# ========================================
# 训练循环
# ========================================

def train_one_epoch(state, loader, batch_size, train_step_fn):
    losses = []
    num_batches = 0

    for batch_images, batch_labels in train_batch_generator(loader, batch_size, shuffle=True):
        state, loss = train_step_fn(state, batch_images, batch_labels)
        losses.append(float(loss))
        num_batches += 1

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return state, avg_loss, num_batches


def validate(state, loader, batch_size, eval_step_fn):
    losses = []
    accuracies = []
    num_samples = 0

    for batch_images, batch_labels in val_batch_generator(loader, batch_size):
        loss, acc = eval_step_fn(
            state.params, state.batch_stats, batch_images, batch_labels
        )
        batch_size_actual = batch_images.shape[0]

        losses.append(float(loss))
        accuracies.append(float(acc))
        num_samples += batch_size_actual

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0

    return avg_loss, avg_acc, num_samples


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mamba",
                        choices=["cnn", 'resnet18', "resnet34", "mamba"])
    parser.add_argument("--data_path", type=str,
                        default=os.path.join("..", "data", "dataset"))
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--train_split_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mamba_patch_size", type=int, default=16)
    parser.add_argument("--mamba_embed_dim", type=int, default=512)
    parser.add_argument("--mamba_use_class_token", type=bool, default=False)
    parser.add_argument("--mamba_dropout_rate", type=float, default=0.1)
    parser.add_argument("--mamba_depth", type=int, default=8)
    parser.add_argument("--mamba_conv_kernel_size", type=int, default=3)

    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    image_size = tuple(args.image_size)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    model_ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    os.makedirs(model_ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model       : {args.model}")
    print(f"Data path   : {os.path.abspath(args.data_path)}")
    print(f"Image size  : {image_size}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Epochs      : {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Checkpoint dir: {os.path.abspath(model_ckpt_dir)}")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] Loading dataset...")
    loader = data_loader(
        data_path=args.data_path,
        image_size=image_size,
        train_split_ratio=args.train_split_ratio,
        use_augmentation=True,
    )

    num_classes = loader.get_num_classes()
    unique_labels = loader.get_unique_labels()
    train_size = len(loader.dataset_train)
    val_size = len(loader.dataset_test)

    print(f"  Classes  : {num_classes}")
    print(f"  Labels   : {unique_labels}")
    print(f"  Train/Val: {train_size} / {val_size}")

    # 创建模型
    print("\n[2/4] Creating model...")
    mamba_config = None
    if args.model == "mamba":
        mamba_config = {
            'patch_size': args.mamba_patch_size,
            'embed_dim': args.mamba_embed_dim,
            'use_class_token': args.mamba_use_class_token,
            'dropout_rate': args.mamba_dropout_rate,
            'depth': args.mamba_depth,
            'conv_kernel_size': args.mamba_conv_kernel_size,
        }

    model = create_model(args.model, num_classes, image_size, mamba_config)

    # 初始化训练状态
    print("\n[3/4] Initializing train state...")
    state = create_train_state(rng, model, args.model, image_size, args.learning_rate)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Total params: {num_params:,}")

    # 创建训练/验证函数
    train_step_fn, eval_step_fn = create_train_and_eval_functions(model, args.model)

    # 训练循环
    print("\n[4/4] Start training...")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(1, args.num_epochs + 1)):
        epoch_start = time.time()

        # 训练
        state, train_loss, n_train_batches = train_one_epoch(
            state, loader, args.batch_size, train_step_fn
        )

        # 验证
        val_loss, val_acc, n_val_samples = validate(
            state, loader, args.batch_size, eval_step_fn
        )

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} ({n_train_batches} batches)")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f} ({n_val_samples} samples)")

        # 保存最佳模型
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

        # 定期保存
        if epoch % args.save_every == 0:
            ckpt = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'epoch': epoch,
            }

            epoch_ckpt_path = os.path.join(model_ckpt_dir, f"epoch_{epoch}.pkl")
            save_checkpoint(epoch_ckpt_path, ckpt)

    print("\n" + "=" * 60)
    print(f"Training done! Best acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print("=" * 60)


if __name__ == "__main__":
    main()
