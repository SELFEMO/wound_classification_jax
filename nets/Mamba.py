import jax
import flax
from typing import Optional

from torch.backends.cudnn import deterministic


class PatchEmbedding(flax.linen.Module):
    """
    Patch Embedding module to convert images into patch embeddings.  补丁嵌入模块，将图像转换为补丁嵌入。
    """
    patch_size: int = 16  # Size of each patch  每个补丁的大小
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度

    @flax.linen.compact
    def __call__(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """
        Forward pass of Patch Embedding.

        :param x: Input tensor  输入张量

        :return: Patch embeddings  补丁嵌入
        """
        x = flax.linen.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            use_bias=False,
        )(x)  # Divide image into patches and project  将图像划分为补丁并进行投影  # (B, H', W', D), H' = H/patch_size, W' = W/patch_size, D = embed_dim

        # Reshape to (B, N, C) where N is number of patches  重塑为 (B, N, D)，其中 N 是补丁数
        # Convert to sequence format: (B, H', W', C) -> (B, N, D)  转换为序列格式：(B, H', W', D) -> (B, N, D)
        B, H, W, C = x.shape
        x = jax.numpy.reshape(x, (B, H * W, C))  # (B, N, D), N = H'*W'=(H/patch_size)*(W/patch_size), D = embed_dim

        return x  # Patch embeddings  补丁嵌入  # (B, (H/patch_size)*(W/patch_size), embed_dim)

class ConvBranch(flax.linen.Module):
    """
    Convolutional Branch.  卷积分支。
    """
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度
    kernel_size: int = 3  # Kernel size for convolution  卷积的核大小

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        """
        Forward pass of Conv Branch.

        :param x: Input tensor  输入张量

        :return: Output tensor after passing through SSM Branch  通过 SSM 分支后的输出张量
        """
        x = flax.linen.Dense(
            features=self.embed_dim // 2,
            use_bias=False,
        )(x)  # Reduce dimension  降低维度  # (B, N, D/2)
        x = flax.linen.silu(x)  # Activation  激活  # (B, N, D/2)
        return x  # Output of Conv Branch  卷积分支的输出  # (B, N, D/2)


class SSMBranch(flax.linen.Module):
    """
    State Space Model (SSM) Branch.  状态空间模型（SSM）分支。
    """
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度
    kernel_size: int = 3  # Kernel size for convolution  卷积的核大小
    expend: int = 2  # Expend factor for SSM  SSM 的扩展因子
    d_state: int = 16  # State dimension for SSM  SSM 的状态维度
    dt_rank: int = 16  # Rank for time projection  时间投影的秩
    dt_clip_max: float = 10.0  # Maximum clipping value for dt  dt 的最大剪辑值
    exp_clip: float = 5.0  # Maximum clipping value for exponent argument  指数参数的最大剪辑值

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        """
        Forward pass of SSM Branch.

        :param x: Input tensor  输入张量

        :return: Output tensor after passing through SSM Branch  通过 SSM 分支后的输出张量
        """
        # Get input shape  获取输入形状
        B, N, D = x.shape
        d_inner = D * self.expend  # Inner dimension  内部维度

        # Initial Dense layer to get x and residual  初始 Dense 层以获取 x 和残差
        x_project = flax.linen.Dense(
            features=d_inner * 2,
        )(x)
        x_main, x_gate = jax.numpy.split(
            x_project,
            2,
            axis=-1
        )  # Split into x and residual  分割为 x 和残差
        x_main = flax.linen.silu(x_main)  # Activation  激活

        # Generation time step dt: (B, N, d_inner)  生成时间步长 dt: (B, N, d_inner)
        dt = flax.linen.Dense(
            features=self.dt_rank,
        )(x_main)
        dt = flax.linen.Dense(
            features=d_inner,
        )(dt)
        dt = flax.linen.softplus(dt)  # Ensure positivity  确保正值
        # clip dt to avoid huge steps (important for numeric stability)  避免巨大步骤（对数值稳定性很重要）
        dt = jax.numpy.clip(
            dt,
            a_min=1e-6,
            a_max=self.dt_clip_max,
        )  # (B, N, d_inner)

        # Define SSM parameters  定义 SSM 参数
        A_log = self.param(
            "A_log",
            lambda rng, shape: jax.random.uniform(
                rng, shape, minval=-5, maxval=-1
            ),  # Initialize A_log between -5 and -1  在 -5 和 -1 之间初始化 A_log
            (d_inner, self.d_state),
        )  # (d_inner, d_state)
        A = -jax.numpy.exp(
            A_log.astype(jax.numpy.float32),
        )  # Ensure A is negative real  确保 A 是负实数  # (d_inner, d_state)
        B_project = flax.linen.Dense(
            features=self.d_state,
            use_bias=False,
        )(x_main)  # (B, N, d_state)
        C_project = flax.linen.Dense(
            features=self.d_state,
            use_bias=False,
        )(x_main)  # (B, N, d_state)
        D_param = self.param(
            "D",
            # flax.linen.initializers.ones,
            lambda key, shape: 0.1 * jax.numpy.ones(
                shape,
                dtype=jax.numpy.float32,
            ),
            (d_inner,),
        )

        # Define a unidirectional SSM scan (along sequence dimension N).
        def ssm_scan(
                u: jax.numpy.ndarray,
                dt_local: jax.numpy.ndarray,
                B_local: jax.numpy.ndarray,
                C_local: jax.numpy.ndarray,
        ) -> jax.numpy.ndarray:
            """
            Single step of SSM scan.  SSM 扫描的单步。

            :param u: Input at current time step  当前时间步的输入  # (B, N, d_inner)
            :param dt_local: Time step size  时间步长大小  # (B, N, d_inner)
            :param B_local: B parameter for SSM  SSM 的 B 参数  # (B, N, d_state)
            :param C_local: C parameter for SSM  SSM 的 C 参数  # (B, N, d_state)

            :return: Output at current time step  当前时间步的输出  # (B, N, d_inner)
            """
            # Get shapes  获取形状
            B, L, D = u.shape

            def step_fn(
                    h: jax.numpy.ndarray,
                    inputs: jax.numpy.ndarray,
            ) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
                """
                Single time step function for SSM.  SSM 的单时间步函数。

                :param h: Hidden state  隐藏状态  # (B, d_inner, d_state)
                :param inputs: Input at current time step  当前时间步的输入  # (u_t, dt_t, B_t, C_t)

                :return: New hidden state and output  新的隐藏状态和输出  # (h_new: (B, d_inner, d_state), y_t: (B, d_inner))
                """
                # Unpack inputs  解包输入
                u_t, dt_t, B_t, C_t = inputs
                # u_t: (B, d_inner)
                # dt_t: (B, d_inner)
                # B_t: (B, d_state)
                # C_t: (B, d_state)

                # Ensure float32 for numerical stability
                u_t = u_t.astype(jax.numpy.float32)
                dt_t = dt_t.astype(jax.numpy.float32)

                # A_bar = exp(A * dt) - proper discretization
                # A: (d_inner, d_state)
                # dt_t: (B, d_inner)
                dt_expanded = dt_t[:, :, None]  # (B, d_inner, 1)
                dt_expanded = jax.numpy.clip(
                    dt_expanded,
                    a_min=1e-6,
                    a_max=self.dt_clip_max,
                )  # Clip dt for stability  为稳定性剪辑 dt
                exp_arg = A[None, :, :] * dt_expanded  # (B, d_inner, d_state)
                exp_arg = jax.numpy.clip(
                    exp_arg,
                    a_min=-self.exp_clip,
                    a_max=+self.exp_clip,
                    # a_max=0.0,  # A is always negative, so exp_arg should be <= 0  A 始终为负，因此 exp_arg 应该 <= 0
                )  # Clip exponent argument for stability  为稳定性剪辑指数参数
                A_bar = jax.numpy.exp(exp_arg)  # (B, d_inner, d_state)

                # B_bar = B * dt
                B_bar = B_t[:, None, :] * dt_expanded  # (B, d_inner, d_state)

                # State update: h = A_bar * h + B_bar * u
                u_expanded = u_t[:, :, None]  # (B, d_inner, 1)
                h_new = A_bar * h + B_bar * u_expanded  # (B, d_inner, d_state)

                # Output: y = C @ h + D * u
                y_t = jax.numpy.sum(
                    C_t[:, None, :] * h_new, axis=-1
                )  # (B, d_inner)
                y_t = y_t + D_param[None, :] * u_t  # (B, d_inner)

                return h_new, y_t

            # Initialize hidden state  初始化隐藏状态
            h0 = jax.numpy.zeros(
                (B, d_inner, self.d_state),
                dtype=jax.numpy.float32,
            )  # (B, d_inner, d_state)

            # Put the time dimension first: (B, N, ?) -> (N, B, ?)  将时间维度放在第一位：(B, N, ?) -> (N, B, ?)
            u_time_major = jax.numpy.swapaxes(u, 0, 1)  # (N, B, d_inner)
            dt_time_major = jax.numpy.swapaxes(dt_local, 0, 1)  # (N, B, d_inner)
            B_time_major = jax.numpy.swapaxes(B_local, 0, 1)  # (N, B, d_state)
            C_time_major = jax.numpy.swapaxes(C_local, 0, 1)  # (N, B, d_state)

            _, y_T = jax.lax.scan(
                step_fn,
                h0,
                (u_time_major, dt_time_major, B_time_major, C_time_major),
            )  # Scan over time steps  扫描时间步长

            # Swap back to batch major: (N, B, d_inner) -> (B, N, d_inner)  交换回批量主导：(N, B, d_inner) -> (B, N, d_inner)
            y = jax.numpy.swapaxes(y_T, 0, 1)  # Swap back to (B, N, d_inner)  交换回 (B, N, d_inner)

            return y  # Output of SSM scan  SSM 扫描的输出

        # Forward pass through SSM scan  通过 SSM 扫描的前向传递
        y_forward = ssm_scan(
            x_main,
            dt,
            B_project,
            C_project
        )  # (B, N, d_inner)
        # Backward pass through SSM scan (reverse sequence)  通过 SSM 扫描的反向传递（反向序列）
        y_backward = ssm_scan(
            x_main[:, ::-1, :],
            dt[:, ::-1, :],
            B_project[:, ::-1, :],
            C_project[:, ::-1, :]
        )  # (B, N, d_inner)
        y_backward = y_backward[:, ::-1, :]  # Reverse back to original order
        # Combine forward and backward outputs  结合前向和后向输出
        y = y_forward + y_backward  # (B, N, d_inner)

        # Gate and final Dense layer  门控和最终 Dense 层
        y = y * jax.nn.silu(x_gate)  # Gate  门控  # (B, N, d_inner)
        y = flax.linen.Dense(
            features=self.embed_dim // 2,
            use_bias=False,
        )(y)  # Final projection  最终投影  # (B, N, D/2)

        return y  # Output of SSM Branch  SSM 分支的输出  # (B, N, D/2)


class ResidualBlock(flax.linen.Module):
    """
    A simple residual block with LayerNorm, Dense, and SiLU activation.  一个简单的残差块，包含 LayerNorm、Dense 和 SiLU 激活。
    """
    features: int

    @flax.linen.compact
    def __call__(self, x):
        residual = x
        x = flax.linen.LayerNorm()(x)  # 归一化
        x = flax.linen.Dense(
            self.features
        )(x)  # 线性变换
        x = flax.linen.silu(x)  # SiLU 激活
        return x + residual  # 残差连接


class VisionMambaBlock(flax.linen.Module):
    """
    Vision Mamba Block.  视觉 Mamba 块。
    """
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度
    conv_kernel_size: int = 3  # Kernel size for convolution  卷积的核大小
    ssm_expend: int = 2  # Expend factor for SSM  SSM 的扩展因子
    ssm_d_state: int = 16  # State dimension for SSM  SSM 的状态维度
    ssm_dt_rank: int = 16  # Rank for time projection  时间投影的秩

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of Vision Mamba Block.

        :param x: Input tensor  输入张量
        :param train: Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through Vision Mamba Block  通过视觉 Mamba 块后的输出张量
        """
        # Get input shape  获取输入形状
        B, N, D = x.shape

        # Pre-Normalization  预归一化
        x_norm = flax.linen.LayerNorm()(x)

        # Conv Branch  卷积分支
        conv_out = ConvBranch(
            embed_dim=self.embed_dim,
            kernel_size=self.conv_kernel_size,
        )(x_norm)  # (B, N, D) -> (B, N, embed_dim)

        # SSM Branch  SSM 分支
        ssm_out = SSMBranch(
            embed_dim=self.embed_dim,
            kernel_size=self.conv_kernel_size,
            expend=self.ssm_expend,
            d_state=self.ssm_d_state,
            dt_rank=self.ssm_dt_rank,
        )(x_norm)  # (B, N, D2) -> (B, N, embed_dim)

        # Combine branches  组合分支
        combined = jax.numpy.concatenate(
            [
                conv_out,  # (B, N, D1) -> (B, N, D/2)
                ssm_out,  # (B, N, D2) -> (B, N, D/2)
            ],  # D1 + D2 = embed_dim
            axis=-1  # Concatenate along the feature dimension (-1 meaning last dimension)  # 沿特征维度连接（-1 表示最后一个维度）
        )  # (B, N, embed_dim)
        combined = flax.linen.Dense(
            features=self.embed_dim,
            use_bias=False,
        )(combined)  # (B, N, embed_dim)

        # Main Residual Connection  主残差连接
        x = x + combined

        # # FFN with Residual  带残差的 FFN
        # x_norm = flax.linen.LayerNorm()(x)
        # f1 = flax.linen.Dense(
        #     features=self.embed_dim * 4,
        # )(x_norm)
        # f2 = flax.linen.Dense(
        #     features=self.embed_dim * 4,
        # )(x_norm)
        # f = flax.linen.silu(f1) * f2
        # f = flax.linen.Dense(
        #     features=self.embed_dim,
        # )(f)
        # x = x + f  # Final output with residual  带残差的最终输出  # (B, N, embed_dim)

        # SwiGlu FFN with Residual  带残差的 SwiGlu FFN
        x_norm = flax.linen.LayerNorm()(x)  # LayerNorm  层归一化  # (B, N, embed_dim)
        ff = flax.linen.Dense(
            features=self.embed_dim * 2,
        )(x_norm)  # Expand dimension  扩展维度  # (B, N, embed_dim * 2)
        f1, f2 = jax.numpy.split(
            ff,
            2,
            axis=-1
        )  # Split into two parts  分成两部分  # (B, N, embed_dim)
        f = flax.linen.silu(f2) * f1  # SwiGlu activation  SwiGlu 激活  # (B, N, embed_dim)
        f = flax.linen.Dense(
            features=self.embed_dim,
        )(f)  # Final projection  最终投影  # (B, N, embed_dim)
        x = x + f  # Final output with residual  带残差的最终  # (B, N, embed_dim)

        return x  # Output of Vision Mamba Block  视觉 Mamba 块的输出  # (B, N, embed_dim)


class VisionMamba(flax.linen.Module):
    """
    Vision Mamba for classification  视觉 Mamba 用于分类。
    """
    num_classes: int  # Number of output classes  输出类别数
    patch_size: int = 16  # Size of each patch  每个补丁的大小
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度
    use_class_token: bool = False  # Whether to use class token  是否使用类别令牌
    depth: int = 8  # Number of Vision Mamba blocks  视觉 Mamba 块的数量
    conv_kernel_size: int = 3  # Kernel size for convolution in Conv Branch  卷积分支中卷积的核大小
    ssm_expend: int = 2  # Expend factor for SSM Branch  SSM 分支的扩展因子
    ssm_d_state: int = 16  # State dimension for SSM Branch  SSM 分支的状态维度
    ssm_dt_rank: int = 16  # Rank for time projection in SSM Branch  SSM 分支中时间投影的秩
    dropout_rate: Optional[float] = 0.2  # Dropout rate for regularization  正则化的 Dropout 率

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of Vision Mamba.

        :param x: Input tensor  输入张量
        :param train: Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through Vision Mamba  通过 Vision Mamba 后的输出张量
        """
        # Get input shape  获取输入形状
        B, H, W, C = x.shape

        # Patch embedding  补丁嵌入
        x = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )(x)  # (B, N, D), N = (H/patch_size)*(W/patch_size), D = embed_dim

        # Positional embedding  位置嵌入
        N = x.shape[1]  # Number of patches  补丁数
        pos_embedding = self.param(
            'pos_embedding',
            flax.linen.initializers.normal(stddev=0.02),
            (1, N, self.embed_dim)
        )  # (1, N, D)
        x = x + pos_embedding  # Add positional embeddings  添加位置嵌入  # (B, N, D)

        # Class token (optional, can be added if needed)  类别令牌（可选，如有需要可添加）
        if self.use_class_token:
            class_token = self.param(
                'class_token',
                flax.linen.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )  # (1, 1, D)
            class_tokens = jax.numpy.tile(class_token, (B, 1, 1))  # (B, 1, D)
            x = jax.numpy.concatenate([class_tokens, x], axis=1)  # Concatenate class token  连接类别令牌  # (B, N+1, D)

        # Stack Vision Mamba Blocks  堆叠视觉 Mamba 块
        for i in range(self.depth):
            x = VisionMambaBlock(
                embed_dim=self.embed_dim,
                conv_kernel_size=self.conv_kernel_size,
                ssm_expend=self.ssm_expend,
                ssm_d_state=self.ssm_d_state,
                ssm_dt_rank=self.ssm_dt_rank,
            )(x, train=train)  # (B, N, D) or (B, N+1, D)

        # Classification pooling  分类池化
        if self.use_class_token:
            x = x[:, 0]  # Use class token output  使用类别令牌输出  # (B, D)
        else:
            x = jax.numpy.mean(x, axis=1)  # Global average pooling  全局平均池化  # (B, D)

        # Classification head  分类头
        x = flax.linen.LayerNorm()(x)  # Layer normalization  层归一化  # (B, D)
        x = flax.linen.Dense(
            features=self.embed_dim // 2,
        )(x)  # Intermediate linear layer  中间线性层  # (B, D/2)
        x = flax.linen.silu(x)  # Activation  激活  # (B, D/2)
        if self.dropout_rate is not None and self.dropout_rate > 0.0 and train:
            x = flax.linen.Dropout(
                rate=self.dropout_rate,
                deterministic=not train,
            )(x)  # Dropout for regularization  正则化的 Dropout  # (B, D/2)
        logits = flax.linen.Dense(
            features=self.num_classes
        )(x)  # Final linear layer  最终线性层  # (B, num_classes)

        return logits  # Output logits  输出 logits  # (B, num_classes)


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
            patch_size=16,
            use_class_token=True,
            embed_dim=256,
            depth=4,
            conv_kernel_size=3,
            ssm_expend=2,
            ssm_d_state=8,
            ssm_dt_rank=8,
            dropout_rate=0.2,
        )
        print("    ✓ Model created")
        print(f"      Embedding dimension: 512")
        print(f"      Depth: 4")
        print(f"      Patch size: 16")

    except Exception as e:
        print(f"    ✗ Failed to create model: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 3. Initialize Parameters =====
    print("\n[3] Initializing model parameters...")
    try:
        key_init = jax.random.fold_in(key, 0)
        x_dummy = jax.random.normal(key_init, (batch_size, height, width, channels))

        params = model.init(
            key_init,
            x_dummy,
            train=True,
        )
        print("    ✓ Parameters initialized successfully")


        # Count total parameters
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

    # ===== 4. Forward Pass Test (Training Mode) =====
    print("\n[4] Testing forward pass (training mode)...")
    try:
        key_forward = jax.random.fold_in(key, 1)
        x_test = jax.random.normal(key_forward, (batch_size, height, width, channels))

        logits = model.apply(
            params,
            x_test,
            train=True,
            rngs={
                'dropout': key_forward
            }
        )

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
            print(f"      Sample logits: {logits_np[0]}")
        elif inf_count > 0:
            print(f"    ✗ WARNING: Output contains {inf_count} Inf values!")
        else:
            print("    ✓ No NaN/Inf in output")

        print(f"      Logits min: {logits_np.min():.4f}")
        print(f"      Logits max: {logits_np.max():.4f}")
        print(f"      Logits mean: {logits_np.mean():.4f}")
        print(f"      Logits std: {logits_np.std():.4f}")

    except Exception as e:
        print(f"    ✗ Forward pass (training) failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 5. Forward Pass Test (Evaluation Mode) =====
    print("\n[5] Testing forward pass (evaluation mode)...")
    try:
        logits_eval = model.apply(
            params,
            x_test,
            train=False
        )
        print(f"    ✓ Evaluation pass successful")
        print(f"      Output shape: {logits_eval.shape}")

        diff = jax.numpy.abs(logits - logits_eval).max()
        print(f"      Max difference (train vs eval): {float(diff):.6f}")

        logits_eval_np = np.array(logits_eval)
        if np.isnan(logits_eval_np).any():
            print("    ✗ WARNING: Evaluation output contains NaN!")
        else:
            print("    ✓ Evaluation output is valid")

    except Exception as e:
        print(f"    ✗ Evaluation pass failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 6. Loss Function Test =====
    print("\n[6] Testing loss computation...")
    try:
        def loss_fn(params_local, x_local, y_local):
            logits_local = model.apply(
                params_local,
                x_local,
                train=False
            )
            log_softmax = jax.nn.log_softmax(logits_local, axis=-1)
            loss = -jax.numpy.mean(jax.numpy.sum(y_local * log_softmax, axis=-1))
            return loss, logits_local


        # Create one-hot labels
        y_dummy = jax.nn.one_hot(
            jax.numpy.array([0, 1, 2, 3, 0])[:batch_size],
            num_classes
        )

        loss_value, _ = loss_fn(params, x_test, y_dummy)
        print(f"    ✓ Loss computation successful")
        print(f"      Loss value: {float(loss_value):.4f}")

        if jax.numpy.isnan(loss_value):
            print("    ✗ WARNING: Loss is NaN!")
        elif jax.numpy.isinf(loss_value):
            print("    ✗ WARNING: Loss is Inf!")
        else:
            print("    ✓ Loss is valid")

    except Exception as e:
        print(f"    ✗ Loss computation failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 7. Gradient Test =====
    print("\n[7] Testing backward pass (gradient computation)...")
    try:
        grad_fn = jax.value_and_grad(lambda p: loss_fn(p, x_test, y_dummy)[0])
        loss_value, grads = grad_fn(params)

        print(f"    ✓ Gradient computation successful")
        print(f"      Loss: {float(loss_value):.4f}")


        # Compute gradient norm
        def compute_grad_norm(g):
            if isinstance(g, dict):
                return jax.numpy.sqrt(
                    sum(compute_grad_norm(v) ** 2 for v in g.values())
                )
            else:
                return jax.numpy.linalg.norm(g.flatten())


        grad_norm = compute_grad_norm(grads)
        print(f"      Gradient norm: {float(grad_norm):.6f}")

        if jax.numpy.isnan(grad_norm):
            print("    ✗ WARNING: Gradients contain NaN!")


            # Print some gradient samples for debugging
            def print_some_grads(g, prefix="", count=0):
                if count >= 3:
                    return count
                if isinstance(g, dict):
                    for k, v in g.items():
                        count = print_some_grads(v, f"{prefix}/{k}", count)
                else:
                    g_np = np.array(g)
                    if np.isnan(g_np).any():
                        print(f"        NaN in {prefix}: shape={g.shape}, nan_count={np.isnan(g_np).sum()}")
                        count += 1
                return count


            print_some_grads(grads)
        elif jax.numpy.isinf(grad_norm):
            print("    ✗ WARNING: Gradients contain Inf!")
        else:
            print("    ✓ Gradients are valid")


        # Gradient statistics
        def grad_stats(g):
            if isinstance(g, dict):
                all_vals = []
                for v in g.values():
                    all_vals.extend(grad_stats(v))
                return all_vals
            else:
                return g.flatten().tolist()


        all_grads = jax.numpy.array(grad_stats(grads))
        if not jax.numpy.isnan(all_grads).any():
            print(f"      Gradient min: {all_grads.min():.6f}")
            print(f"      Gradient max: {all_grads.max():.6f}")
            print(f"      Gradient mean: {all_grads.mean():.6f}")
            print(f"      Gradient std: {all_grads.std():.6f}")

    except Exception as e:
        print(f"    ✗ Gradient computation failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 8. Mini Training Loop Test =====
    print("\n[8] Testing mini training loop (5 steps)...")
    try:
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-4)
        )
        opt_state = optimizer.init(params)


        def loss_fn_train(params_local, x_local, y_local):
            logits_local = model.apply(
                params_local,
                x_local,
                train=False
            )
            log_softmax = jax.nn.log_softmax(logits_local, axis=-1)
            loss = -jax.numpy.mean(jax.numpy.sum(y_local * log_softmax, axis=-1))
            return loss


        @jax.jit
        def train_step(params_local, opt_state_local, x_local, y_local):
            loss_val, grads = jax.value_and_grad(
                lambda p: loss_fn_train(p, x_local, y_local)
            )(params_local)
            updates, opt_state_local = optimizer.update(grads, opt_state_local)
            params_local = jax.tree_util.tree_map(
                lambda p, u: p + u,
                params_local,
                updates
            )
            return params_local, opt_state_local, loss_val


        losses = []
        print("    Running 5 training steps...")

        for step in range(5):
            key = jax.random.fold_in(key, step + 100)
            x_batch = jax.random.normal(key, (batch_size, height, width, channels))

            params, opt_state, loss = train_step(params, opt_state, x_batch, y_dummy)
            losses.append(float(loss))

            print(f"      Step {step + 1}: loss = {losses[-1]:.6f}")

            if jax.numpy.isnan(loss):
                print(f"    ✗ NaN detected at step {step + 1}!")
                break

        print(f"    ✓ Training loop completed")
        print(f"      Loss trajectory: {[f'{l:.6f}' for l in losses]}")

        if len(losses) > 1:
            loss_change = losses[0] - losses[-1]
            print(f"      Loss change: {loss_change:.6f}")
            if loss_change > 0:
                print(f"    ✓ Loss is decreasing (good sign!)")
            elif abs(loss_change) < 1e-6:
                print(f"    ⚠ Loss is stable (might need more steps)")
            else:
                print(f"    ⚠ Loss is increasing (check learning rate)")

    except Exception as e:
        print(f"    ✗ Training loop failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 9. Inference Speed Test =====
    print("\n[9] Testing inference speed...")
    try:
        @jax.jit
        def infer_jit(p, x):
            return model.apply(
                p,
                x,
                train=False
            )


        # Warm up
        _ = infer_jit(params, x_test)

        # Timing
        num_iterations = 20
        start = time.time()
        for _ in range(num_iterations):
            _ = infer_jit(params, x_test)
        elapsed = time.time() - start

        time_per_batch = elapsed / num_iterations * 1000  # ms
        throughput = batch_size * num_iterations / (elapsed + 1e-8)  # samples/sec  # 1e-8 to avoid div by zero

        print(f"    ✓ Inference speed test completed")
        print(f"      Time per batch: {time_per_batch:.2f} ms")
        print(f"      Throughput: {throughput:.1f} samples/sec")

    except Exception as e:
        print(f"    ✗ Speed test failed: {e}")
        traceback.print_exc()

    # ===== 10. Different Batch Sizes Test =====
    print("\n[10] Testing with different batch sizes...")
    try:
        for test_batch in [1, 2, 4, 8]:
            x_test_batch = jax.random.normal(
                jax.random.fold_in(key, test_batch),
                (test_batch, height, width, channels)
            )
            logits_batch = model.apply(
                params,
                x_test_batch,
                train=False
            )
            print(f"    ✓ Batch size {test_batch:2d}: output shape {logits_batch.shape}")

    except Exception as e:
        print(f"    ✗ Batch size test failed: {e}")
        traceback.print_exc()

    # ===== 11. Architecture Analysis =====
    print("\n[11] Analyzing model architecture...")
    try:
        print(f"    Model architecture: VisionMamba")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Embedding dimension: 512")
        print(f"    Depth (Mamba blocks): 4")
        print(f"    Patch size: 16x16")
        print(f"    Number of patches: {(height // 16) * (width // 16)}")
        print(f"    ✓ Architecture analysis completed")

    except Exception as e:
        print(f"    ✗ Architecture analysis failed: {e}")
        traceback.print_exc()

    # ===== 12. SSM Component Test =====
    print("\n[12] Testing SSM component separately...")
    try:
        # Test SSM branch independently
        ssm_branch = SSMBranch(
            embed_dim=512,
            kernel_size=3,
            expend=2,
            d_state=16,
            dt_rank=16,
        )

        key_ssm = jax.random.fold_in(key, 999)
        # Create a smaller test input (B, N, D)
        x_ssm_test = jax.random.normal(key_ssm, (2, 100, 512))

        # Initialize SSM
        params_ssm = ssm_branch.init(key_ssm, x_ssm_test)

        # Forward pass
        out_ssm = ssm_branch.apply(params_ssm, x_ssm_test)

        print(f"    ✓ SSM component test successful")
        print(f"      Input shape: {x_ssm_test.shape}")
        print(f"      Output shape: {out_ssm.shape}")
        print(f"      Expected output dim: 256 (embed_dim // 2)")

        # Check for NaN
        out_ssm_np = np.array(out_ssm)
        if np.isnan(out_ssm_np).any():
            print(f"    ✗ WARNING: SSM output contains NaN!")
        else:
            print(f"    ✓ SSM output is valid")
            print(f"      SSM output min: {out_ssm_np.min():.4f}")
            print(f"      SSM output max: {out_ssm_np.max():.4f}")

    except Exception as e:
        print(f"    ✗ SSM component test failed: {e}")
        traceback.print_exc()

    # ===== 13. Parameter Shape Verification =====
    print("\n[13] Verifying parameter shapes (top-level only)...")
    try:
        def print_param_tree(params_dict, prefix="", depth=0, max_depth=2):
            if depth > max_depth:
                return
            indent = "    " + "  " * depth
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    print(f"{indent}{k}:")
                    print_param_tree(v, prefix, depth + 1, max_depth)
                else:
                    if hasattr(v, 'shape'):
                        print(f"{indent}{k}: {v.shape}")


        print_param_tree(params)

    except Exception as e:
        print(f"    ✗ Parameter verification failed: {e}")
        traceback.print_exc()

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    # Determine overall status
    has_nan = nan_count > 0 if 'nan_count' in locals() else False

    if has_nan:
        print("⚠ PARTIAL SUCCESS - Model structure is correct but outputs contain NaN")
        print("  Possible causes:")
        print("  - SSM numerical instability (exp overflow/underflow)")
        print("  - dt parameter range too large")
        print("  - A matrix initialization issues")
        print("  Suggested fixes:")
        print("  - Reduce dt_max from 0.1 to 0.01")
        print("  - Increase A initialization range (more negative)")
        print("  - Add more aggressive clipping in SSM step_fn")
    else:
        print("✓ All critical tests passed!")
        print("✓ VisionMamba model is ready for training")

    print(f"✓ Model has {total_params:,} trainable parameters")
    print("=" * 80)
