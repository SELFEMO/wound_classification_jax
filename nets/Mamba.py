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
        )(x)  # Divide image into patches and project  将图像划分为补丁并进行投影

        # Reshape to (B, N, C) where N is number of patches  重塑为 (B, N, C)，其中 N 是补丁数
        # Convert to sequence format: (B, H', W', C) -> (B, N, D)  转换为序列格式：(B, H', W', C) -> (B, N, D)
        B, H, W, C = x.shape
        x = jax.numpy.reshape(x, (B, H * W, C))  #

        return x  # Patch embeddings  补丁嵌入


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
        x = flax.linen.Conv(
            features=self.embed_dim // 2,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            feature_group_count=self.embed_dim // 2,
            use_bias=False,
        )(x)  # Depthwise convolution  深度卷积  # (B, N, D/2)
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
        x_and_res = flax.linen.Dense(
            features=d_inner * 2,
        )(x)
        x_, res = jax.numpy.split(
            x_and_res,
            2,
            axis=-1
        )  # Split into x and residual  分割为 x 和残差
        x_ = flax.linen.silu(x_)  # Activation  激活
        res = flax.linen.Dense(
            features=self.embed_dim // 2,
        )(res)  # Project residual back to embed_dim/2  将残差投影回 embed_dim/2

        # Generation time step dt: (B, N, d_inner)  生成时间步长 dt: (B, N, d_inner)
        dt = flax.linen.Dense(
            features=self.dt_rank,
        )(x_)
        dt = flax.linen.Dense(
            features=d_inner,
        )(dt)
        dt = flax.linen.softplus(dt)  # Ensure positivity  确保正值

        # Define SSM parameters (shared per channel, does not change over time)  定义 SSM 参数（每个通道共享，随时间不变）
        # A, B, C, D_param all have shape (d_inner, d_state) or (d_inner, 1) -> (d_inner,)  A、B、C、D_param 全部具有形状 (d_inner, d_state) 或 (d_inner, 1) -> (d_inner，)
        A = self.param(
            "A",
            lambda key, shape: -jax.numpy.ones(shape),  # 初始化为负，产生衰减
            (d_inner,),
        )
        B_param = self.param(
            "B",
            flax.linen.initializers.ones,
            (d_inner,),
        )
        C_param = self.param(
            "C",
            flax.linen.initializers.ones,
            (d_inner,),
        )
        D_param = self.param(
            "D",
            flax.linen.initializers.zeros,
            (d_inner,),
        )

        # Define a unidirectional SSM scan (along sequence dimension N).
        def ssm_scan(
                u: jax.numpy.ndarray,
                dt: jax.numpy.ndarray,
        ) -> jax.numpy.ndarray:
            """
            Single step of SSM scan.  SSM 扫描的单步。

            :param u: Input at current time step  当前时间步的输入  # (B, N, d_inner)
            :param dt: Time step size  时间步长大小  # (B, N, d_inner)

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

                :param h: Hidden state  隐藏状态  # (B, d_inner,)
                :param inputs: Input at current time step  当前时间步的输入  # (B, d_inner,)

                :return: New hidden state and output  新的隐藏状态和输出  # (B, d_inner,), (B, d_inner,)
                """
                # Unpack inputs  解包输入
                u_t, dt_t = inputs

                # Decay factor: exp(A * dt)  衰减因子：exp(A * dt)
                decay = jax.numpy.exp(A * dt_t)  # (B, d_inner)
                # Update hidden state  更新隐藏状态
                h_new = h * decay + B_param * u_t * dt_t  # (B, d_inner)
                # Compute output  计算输出
                y_t = C_param * h_new + D_param * u_t  # (B, d_inner)

                return h_new, y_t

            # Initialize hidden state  初始化隐藏状态
            h0 = jax.numpy.zeros((B, d_inner))  # (B, d_inner)

            # Put the time dimension first: (B, N, d_inner) -> (N, B, d_inner)  将时间维度放在第一位：(B, N, d_inner) -> (N, B, d_inner)
            u_time_major = jax.numpy.swapaxes(u, 0, 1)  # (N, B, d_inner)
            dt_time_major = jax.numpy.swapaxes(dt, 0, 1)  # (N, B, d_inner)

            _, y_T = jax.lax.scan(
                step_fn,
                h0,
                (u_time_major, dt_time_major),
            )  # Scan over time steps  扫描时间步长

            # Swap back to batch major: (N, B, d_inner) -> (B, N, d_inner)  交换回批量主导：(N, B, d_inner) -> (B, N, d_inner)
            y = jax.numpy.swapaxes(y_T, 0, 1)  # Swap back to (B, N, d_inner)  交换回 (B, N, d_inner)

            return y  # Output of SSM scan  SSM 扫描的输出

        # Forward pass through SSM scan  通过 SSM 扫描的前向传递
        y_forward = ssm_scan(x_, dt)  # (B, N, d_inner)
        # Backward pass through SSM scan (reverse sequence)  通过 SSM 扫描的反向传递（反向序列）
        y_backward = ssm_scan(x_[:, ::-1, :], dt[:, ::-1, :])  # (B, N, d_inner)
        y_backward = y_backward[:, ::-1, :]  # Reverse back to original order
        # Combine forward and backward outputs  结合前向和后向输出
        y = y_forward + y_backward  # (B, N, d_inner)

        # Gateed output  门控输出
        gated = x_ * y  # (B, N, d_inner)

        # Final Dense layer to project back to embed_dim/2  最终 Dense 层投影回 embed_dim/2
        out = flax.linen.Dense(
            features=self.embed_dim // 2,
        )(gated)  # (B, N, D)

        # Add residual  添加残差
        out = out + res  # (B, N, D/2)

        return out  # Output of SSM Branch  SSM 分支的输出  # (B, N, D/2)


class ResidualBlock(flax.linen.Module):
    """
    A simple residual block with LayerNorm, Dense, and SiLU activation.  一个简单的残差块，包含 LayerNorm、Dense 和 SiLU 激活。
    """
    features: int

    @flax.linen.compact
    def __call__(self, x):
        residual = x
        x = flax.linen.LayerNorm(

        )(x)  # 归一化
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

        # # Pre-Normalization  预归一化
        # x_norm = flax.linen.LayerNorm()(x)

        # Conv Branch  卷积分支
        conv_out = ConvBranch(
            embed_dim=self.embed_dim,
            kernel_size=self.conv_kernel_size,
            # )(x_norm)
        )(x)  # (B, N, D)

        # SSM Branch  SSM 分支
        ssm_out = SSMBranch(
            embed_dim=self.embed_dim,
            kernel_size=self.conv_kernel_size,
            expend=self.ssm_expend,
            d_state=self.ssm_d_state,
            dt_rank=self.ssm_dt_rank,
            # )(x_norm)
        )(x)  # (B, N, D)

        # Combine branches  组合分支
        combined = jax.numpy.concatenate([conv_out, ssm_out], axis=-1)  # (B, N, D)

        # # Linear projection to embed_dim  线性投影到 embed_dim
        # combined = flax.linen.Dense(
        #     features=self.embed_dim,
        #     use_bias=False,
        # )(combined)
        # # Residual connection  残差连接
        # x = x + combined
        x = ResidualBlock(
            features=self.embed_dim
        )(x)

        return x  # Output of Vision Mamba Block  视觉 Mamba 块的输出


class VisionMamba(flax.linen.Module):
    """
    Vision Mamba for classification  视觉 Mamba 用于分类。
    """
    num_classes: int  # Number of output classes  输出类别数
    patch_size: int = 16  # Size of each patch  每个补丁的大小
    embed_dim: int = 512  # Dimension of the embedding  嵌入的维度
    use_class_token: bool = False  # Whether to use class token  是否使用类别令牌
    dropout_rate: Optional[float] = None  # Dropout rate after embeddings  嵌入后的 dropout 率
    depth: int = 8  # Number of Vision Mamba blocks  视觉 Mamba 块的数量
    conv_kernel_size: int = 3  # Kernel size for convolution in Conv Branch  卷积分支中卷积的核大小
    ssm_expend: int = 2  # Expend factor for SSM Branch  SSM 分支的扩展因子
    ssm_d_state: int = 16  # State dimension for SSM Branch  SSM 分支的状态维度
    ssm_dt_rank: int = 16  # Rank for time projection in SSM Branch  SSM 分支中时间投影的秩

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
        )(x)  # (B, N, D), N = (H/patch_size)*(W/patch_size)

        # Positional embedding  位置嵌入
        N = x.shape[1]  # Number of patches  补丁数
        pos_embedding = self.param(
            'pos_embedding',
            flax.linen.initializers.normal(stddev=0.02),
            (1, N, self.embed_dim)
        )
        x = x + pos_embedding  # Add positional embeddings  添加位置嵌入

        # Class token (optional, can be added if needed)  类别令牌（可选，如有需要可添加）
        if self.use_class_token:
            class_token = self.param(
                'class_token',
                flax.linen.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )
            class_tokens = jax.numpy.tile(class_token, (B, 1, 1))  # (B, 1, D)
            x = jax.numpy.concatenate([class_tokens, x], axis=1)  # (B, N+1, D)  # Concatenate class token  连接类别令牌

        # Dropout (optional) after embeddings  dropout（可选）在嵌入后
        if self.dropout_rate is not None:
            x = flax.linen.Dropout(
                rate=self.dropout_rate,
                deterministic=not train
            )(x)

        # Stack Vision Mamba Blocks  堆叠视觉 Mamba 块
        for i in range(self.depth):
            x = VisionMambaBlock(
                embed_dim=self.embed_dim,
                conv_kernel_size=self.conv_kernel_size,
                ssm_expend=self.ssm_expend,
                ssm_d_state=self.ssm_d_state,
                ssm_dt_rank=self.ssm_dt_rank,
            )(x, train=train)

        # Classification pooling  分类池化
        if self.use_class_token:
            x = x[:, 0]  # Use class token output  使用类别令牌输出
        else:
            x = jax.numpy.mean(x, axis=1)  # Global average pooling  全局平均池化

        # Classification head  分类头
        x = flax.linen.LayerNorm()(x)  # Layer normalization  层归一化
        if self.dropout_rate is not None:
            x = flax.linen.Dropout(
                rate=self.dropout_rate,
                deterministic=not train
            )(x)  # Dropout before final layer  最终层前的 dropout
        logits = flax.linen.Dense(
            features=self.num_classes
        )(x)  # Final linear layer  最终线性层

        return logits  # Output logits  输出 logits


# ===== Test Code =====
if __name__ == "__main__":
    # Select device  选择设备
    device = jax.devices()[0]
    print(f"Using device: {device}")

    # Input parameters  输入参数
    key = jax.random.PRNGKey(0)
    batch_size = 4
    height, width, channels = 224, 224, 3
    num_classes = 5

    # Random input tensor  随机输入张量
    x = jax.random.normal(key, (batch_size, height, width, channels))

    # Build model  构建模型
    model = VisionMamba(
        num_classes=num_classes,
        patch_size=16,
        embed_dim=512,
        use_class_token=False,
        dropout_rate=0.1,
        depth=8,
        conv_kernel_size=3,
    )

    # Initialize parameters  初始化参数
    params = model.init(key, x)

    # Forward pass  前向传播
    logits = model.apply(params, x, train=False)

    print("VisionMamba logits shape:", logits.shape)  # Predicted logits shape: (4, 5)  预测的 logits 形状：(4, 5)
