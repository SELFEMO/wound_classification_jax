import jax
import flax

class basic_block(flax.linen.Module):
    """
    A basic block for ResNet18/34 .  ResNet18/34 的基本模块。

    Conv(out_channels, stride) -> BatchNorm -> ReLU ->
    Conv(out_channels, 1) -> BatchNorm -> Add(skip connection) -> ReLU
    """
    out_channels: int
    strides: tuple = (1, 1)
    use_projection: bool = False


    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the basic block.

        :param x: Input tensor  输入张量
        :param train:  Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through the basic block  通过基本模块后的输出张量
        """
        # Save input for the skip connection  保存输入以进行跳跃连接
        residual = x

        # Convolutional layer 1  卷积层 1
        x = flax.linen.Conv(
            features=self.out_channels,  # Number of output channels  输出通道数
            kernel_size=(3, 3),  # The size of the convolutional kernel  卷积核的大小
            strides=self.strides,  # The stride of the convolution  卷积的步幅
            padding='SAME',  # Padding method  填充方法
            use_bias=False,  # No bias for Conv when using BatchNorm  使用批量归一化时，Conv 不使用偏置
        )(x)
        x = flax.linen.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
        )(x)
        x = flax.linen.relu(x)

        # Convolutional layer 2  卷积层 2
        x = flax.linen.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=False,
        )(x)
        x = flax.linen.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
        )(x)

        # Projection for the skip connection if needed  如果需要，为跳跃连接进行投影
        if self.use_projection:
            residual = flax.linen.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                strides=self.strides,
                padding='SAME',
                use_bias=False,
            )(residual)
            residual = flax.linen.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
            )(residual)

        # Add skip connection  添加跳跃连接
        x += residual
        y = flax.linen.relu(x)
        return y  # Output tensor  输出张量


class ResNet18(flax.linen.Module):
    """
    ResNet18 model.  ResNet18 模型。

    Conv(out_channels, stride) -> BatchNorm -> ReLU ->MaxPool(3x3, 2) ->
    [basic_block * 2] x 4 ->
    GlobalAvgPool -> FullyConnected(num_classes)
    """
    num_classes: int = 2  # Number of output classes  输出类别数


    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the ResNet18 model.

        :param x: Input tensor  输入张量
        :param train:  Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through the ResNet18 model  通过 ResNet18 模型后的输出张量
        """
        # Initial convolutional layer  初始卷积层
        x = flax.linen.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
        )(x)
        x = flax.linen.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
        )(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding='SAME',
        )

        # Residual blocks configuration  残差块配置
        block_configs = [
            (64, 2, False),   # (out_channels, num_blocks, use_projection)  （输出通道数，块数，使用投影）
            (128, 2, True),
            (256, 2, True),
            (512, 2, True),
        ]

        # Build residual blocks  构建残差块
        for out_channels, num_blocks, use_projection in block_configs:
            for i in range(num_blocks):
                x = basic_block(
                    out_channels=out_channels,  # Number of output channels  输出通道数
                    strides=(2, 2) if i == 0 and use_projection else (1, 1),  # Stride for the first block if projection is used  如果使用投影，则为第一个块的步幅
                    use_projection=(i == 0 and use_projection),  # Use projection for the first block if specified  如果指定，则为第一个块使用投影
                )(x, train=train)

        # Global average pooling  全局平均池化
        x = jax.numpy.mean(x, axis=(1, 2))

        # Fully connected layer  全连接层
        y = flax.linen.Dense(
            features=self.num_classes
        )(x)

        return y  # Output tensor  输出张量

# ===== Test Code =====
if __name__ == "__main__":
    # Input parameters  输入参数
    key = jax.random.PRNGKey(0)
    batch_size = 4
    height, width, channels = 224, 224, 3
    num_classes = 5

    # Random input tensor  随机输入张量
    x = jax.random.normal(key, (batch_size, height, width, channels))

    # Build model  构建模型
    model = ResNet18(num_classes=num_classes)

    # Initialize parameters  初始化参数
    params = model.init(key, x)

    # Forward pass  前向传播
    logits = model.apply(params, x, train=False)

    print("ResNet18 logits shape:", logits.shape)  # Predicted logits shape: (4, 5)  预测的 logits 形状：(4, 5)
