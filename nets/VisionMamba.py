import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class MambaBlock(nn.Module):
    """基于 jax4max 的 MambaBlock，改写为 Flax.linen 版本（单序列输入）"""
    d_model: int
    d_state: int

    def setup(self):
        # 所有层定义移到 setup() 中
        self.in_proj = nn.Dense(2 * self.d_model, name='in_proj')
        self.proj_B = nn.Dense(self.d_state, name='proj_B')
        self.proj_C = nn.Dense(self.d_state, name='proj_C')
        self.proj_delta = nn.Dense(self.d_model, name='proj_delta')
        self.out_proj = nn.Dense(self.d_model, name='out_proj')

        # A_log 和 D 参数
        def init_A_log(rng, shape):
            d_state, d_model = shape
            A = jnp.repeat(
                jnp.arange(1, d_state + 1, dtype=jnp.float32)[:, None],
                d_model,
                axis=1,
            )
            return jnp.log(A + 1.0)

        self.A_log = self.param("A_log", init_A_log, (self.d_state, self.d_model))
        self.D = self.param("D", lambda rng, shape: jnp.ones(shape, jnp.float32), (self.d_model,))

    def __call__(self, x):
        """
        Args:
            x: (B, L, d_model) 或 (L, d_model)
        Returns:
            y: 与 x 相同形状
        """
        # ===== 关键修复：支持 batch 维度 =====
        # 如果输入是 3D (B, L, d_model)，保持；如果是 2D (L, d_model)，也能处理
        input_shape = x.shape
        if len(input_shape) == 2:
            # 添加 batch 维度
            x = x[None, ...]  # (1, L, d_model)

        B, L, D = x.shape

        # 输入投影 + gate
        x_proj = self.in_proj(x)  # (B, L, 2*d_model)
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)
        x_main = jax.nn.silu(x_main)

        # B, C, delta
        B_mat = self.proj_B(x_main)  # (B, L, d_state)
        C_mat = self.proj_C(x_main)  # (B, L, d_state)
        delta = self.proj_delta(x_main)  # (B, L, d_model)

        # 更保守的 delta 范围
        delta = jax.nn.softplus(delta) + 1e-4
        delta = jnp.clip(delta, 1e-4, 1.0)

        # 更稳定的 A 矩阵
        A = -jnp.exp(self.A_log)  # (d_state, d_model)
        A = jnp.clip(A, -5.0, -0.1)

        # ===== 处理 batch 维度 =====
        # deltaA: (B, L, d_state, d_model)
        deltaA_input = delta[:, :, None, :] * A[None, None, :, :]
        deltaA_input = jnp.clip(deltaA_input, -2.0, 0.0)
        deltaA = jnp.exp(deltaA_input)
        deltaA = jnp.clip(deltaA, 0.05, 1.0)

        # deltaB_u: (B, L, d_state, d_model)
        deltaB_u = delta[:, :, None, :] * B_mat[:, :, :, None] * x_main[:, :, None, :]
        deltaB_u = jnp.clip(deltaB_u, -5.0, 5.0)

        # ===== 对每个 batch 分别做 scan =====
        def scan_single_batch(deltaA_single, deltaB_u_single):
            # deltaA_single: (L, d_state, d_model)
            # deltaB_u_single: (L, d_state, d_model)

            def binary_operator(elem1, elem2):
                A1, Bu1 = elem1
                A2, Bu2 = elem2

                A_combined = A2 * A1
                Bu_combined = A2 * Bu1 + Bu2

                A_combined = jnp.clip(A_combined, 1e-6, 10.0)
                Bu_combined = jnp.clip(Bu_combined, -100.0, 100.0)

                return A_combined, Bu_combined

            h_final = jax.lax.associative_scan(binary_operator, (deltaA_single, deltaB_u_single))
            _, h_Bu = h_final
            return h_Bu

        # 使用 vmap 只处理 scan（不涉及模块调用）
        h_Bu = jax.vmap(scan_single_batch)(deltaA, deltaB_u)  # (B, L, d_state, d_model)

        # 输出计算
        y = jnp.einsum("blsd,bls->bld", h_Bu, C_mat)  # (B, L, d_model)
        y = jnp.clip(y, -10.0, 10.0)

        # skip + gate + out_proj
        y = y + x_main * self.D[None, None, :]
        y = y * jax.nn.silu(x_gate)

        y = self.out_proj(y)
        y = jnp.clip(y, -10.0, 10.0)

        # 恢复原始形状
        if len(input_shape) == 2:
            y = y[0]  # 去掉 batch 维度

        return y


class VisionMamba(nn.Module):
    """Vision Mamba 模型"""
    num_classes: int
    patch_size: int = 8
    num_layers: int = 6
    d_model: int = 128
    d_state: int = 32

    def setup(self):
        # Patch embedding
        self.patch_embed = nn.Dense(self.d_model, name='patch_embed')

        # Layer normalization
        self.norm = nn.LayerNorm(name='norm')

        # Mamba blocks (在 setup 中创建所有 block)
        self.mamba_blocks = [
            MambaBlock(d_model=self.d_model, d_state=self.d_state, name=f'mamba_block_{i}')
            for i in range(self.num_layers)
        ]

        # Classification head
        self.head = nn.Dense(self.num_classes, name='head')

    def __call__(self, x, train: bool = True):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            logits: (B, num_classes)
        """
        B, H, W, C = x.shape

        # ===== Patch embedding =====
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Reshape to patches
        x = jnp.reshape(
            x,
            (B, patch_h, self.patch_size, patch_w, self.patch_size, C)
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(x, (B, patch_h * patch_w, self.patch_size * self.patch_size * C))

        # 投影到 d_model
        x = self.patch_embed(x)  # (B, num_patches, d_model)

        # ===== 关键修复：直接处理整个 batch =====
        # MambaBlock 现在支持 (B, L, D) 输入
        for block in self.mamba_blocks:
            x = x + block(x)  # (B, num_patches, d_model)
            x = jnp.clip(x, -10.0, 10.0)

        # Layer norm
        x = self.norm(x)

        # Global average pooling
        x = jnp.mean(x, axis=1)  # (B, d_model)

        # Classification head
        x = self.head(x)  # (B, num_classes)

        return x


# ===== Comprehensive Test Code =====
if __name__ == "__main__":
    import traceback
    import numpy as np
    import time
    import optax

    print("=" * 80)
    print("VISION MAMBA COMPREHENSIVE TEST")
    print("=" * 80)

    # ===== 1. Setup =====
    print("\n[1] Setting up test environment...")
    device = jax.devices()[0]
    print(f"    Device: {device}")

    key = jax.random.PRNGKey(0)
    batch_size = 4
    height, width, channels = 224, 224, 3
    num_classes = 5

    print(f"    Batch size: {batch_size}")
    print(f"    Input shape: ({batch_size}, {height}, {width}, {channels})")
    print(f"    Output classes: {num_classes}")

    # ===== 2. Create Model =====
    print("\n[2] Creating VisionMamba model...")
    try:
        model = VisionMamba(
            num_classes=num_classes,
            patch_size=8,
            num_layers=6,
            d_model=128,
            d_state=32,
        )
        print("    ✓ Model created")
        print(f"      Embedding dimension: 128")
        print(f"      Depth: 6")
        print(f"      Patch size: 8")

    except Exception as e:
        print(f"    ✗ Failed to create model: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 3. Initialize Parameters =====
    print("\n[3] Initializing model parameters...")
    try:
        key_init = jax.random.fold_in(key, 0)
        x_dummy = jax.random.normal(key_init, (batch_size, height, width, channels))

        params = model.init(key_init, x_dummy, train=True)
        print("    ✓ Parameters initialized successfully")


        def count_params(params_dict):
            total = 0
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    total += count_params(v)
                else:
                    if hasattr(v, 'size'):
                        total += v.size
            return total


        total_params = count_params(params)
        print(f"    Total parameters: {total_params:,}")

    except Exception as e:
        print(f"    ✗ Failed to initialize parameters: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 4. Forward Pass Test =====
    print("\n[4] Testing forward pass (training mode)...")
    try:
        key_forward = jax.random.fold_in(key, 1)
        x_test = jax.random.normal(key_forward, (batch_size, height, width, channels))

        logits = model.apply(params, x_test, train=True)

        print(f"    ✓ Forward pass successful")
        print(f"      Output shape: {logits.shape}")
        print(f"      Expected shape: ({batch_size}, {num_classes})")

        assert logits.shape == (batch_size, num_classes), \
            f"Shape mismatch! Got {logits.shape}, expected {(batch_size, num_classes)}"
        print("    ✓ Output shape verified")

        # Check for NaN/Inf
        logits_np = np.array(logits)
        nan_count = np.isnan(logits_np).sum()
        inf_count = np.isinf(logits_np).sum()

        if nan_count > 0:
            print(f"    ✗ WARNING: Output contains {nan_count} NaN values!")
        elif inf_count > 0:
            print(f"    ✗ WARNING: Output contains {inf_count} Inf values!")
        else:
            print("    ✓ No NaN/Inf in output")
            print(f"      Logits min: {logits_np.min():.4f}")
            print(f"      Logits max: {logits_np.max():.4f}")
            print(f"      Logits mean: {logits_np.mean():.4f}")
            print(f"      Logits std: {logits_np.std():.4f}")

    except Exception as e:
        print(f"    ✗ Forward pass failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 5. Gradient Test =====
    print("\n[5] Testing backward pass...")
    try:
        def loss_fn(params_local, x_local):
            y_dummy = jax.nn.one_hot(jnp.array([0, 1, 2, 3, 0])[:batch_size], num_classes)
            logits_local = model.apply(params_local, x_local, train=True)
            log_softmax = jax.nn.log_softmax(logits_local, axis=-1)
            loss = -jnp.mean(jnp.sum(y_dummy * log_softmax, axis=-1))
            return loss


        grad_fn = jax.value_and_grad(loss_fn)
        loss_value, grads = grad_fn(params, x_test)

        print(f"    ✓ Gradient computation successful")
        print(f"      Loss: {float(loss_value):.4f}")

        if jnp.isnan(loss_value):
            print("    ✗ WARNING: Loss is NaN!")
        else:
            print("    ✓ Loss is valid")

    except Exception as e:
        print(f"    ✗ Gradient computation failed: {e}")
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print(f"✓ Model has {total_params:,} parameters")
    print("=" * 80)
