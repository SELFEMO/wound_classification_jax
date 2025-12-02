import jax
import flax
from typing import Optional


class SimpleCNN(flax.linen.Module):
    """
    A simple Convolutional Neural Network.  简单卷积神经网络。

    Conv(32) -> BatchNorm -> ReLU -> MaxPool ->
    Conv(64) -> BatchNorm -> ReLU -> MaxPool ->
    Conv(128) -> BatchNorm -> ReLU -> GAP -> Dense(num_classes)
    """
    num_classes: int

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray, train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the CNN.

        :param x: Input tensor  输入张量
        :param train:  Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through the CNN  通过 CNN 后的输出张量
        """
        # x: (B, H, W, C)

        # Block 1
        x = flax.linen.Conv(
            features=32,  # Number of output channels  输出通道数
            kernel_size=(3, 3),  # The size of the convolutional kernel  卷积核的大小
            strides=(1, 1),  # The stride of the convolution  卷积的步幅
            padding='SAME',  # Padding method  填充方法
            use_bias=False,  # No bias for Conv when using BatchNorm  使用批量归一化时，Conv 不使用偏置
        )(x)  # Convolutional layer  卷积层
        x = flax.linen.BatchNorm(
            use_running_average=not train,  # Use running average during evaluation  在评估期间使用运行平均值
            momentum=0.9,  # Momentum for the moving average  移动平均的动量
            epsilon=1e-5,  # Small constant to avoid division by zero  避免除以零的小常数
        )(x)  # Batch normalization layer  批量归一化层
        x = flax.linen.relu(x)  # Activation function  激活函数
        x = flax.linen.max_pool(
            x,  # Input tensor  输入张量
            window_shape=(2, 2),  # Pooling window size  池化窗口大小
            strides=(2, 2),  # Pooling stride  池化步幅
            padding='VALID',  # Padding method  填充方法
        )  # Max pooling layer  最大池化层

        # Block 2
        x = flax.linen.Conv(
            features=64,
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
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(
            x,
            window_shape=(2, 2),
            strides=(2, 2),
            padding='VALID',
        )

        # Block 3
        x = flax.linen.Conv(
            features=128,
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
        x = flax.linen.relu(x)

        # (B, H, W, C) -> (B, C)
        x = jax.numpy.mean(x, axis=(1, 2))  # Global Average Pooling  全局平均池化

        # # Hidden Dense layer  隐藏全连接层
        # x = flax.linen.Dense(
        #     features=256,  # Number of hidden units  隐藏单元数
        # )(x)
        # x = flax.linen.relu(x)  # Activation function  激活函数

        # Output layer: Classification  输出层：分类
        y = flax.linen.Dense(
            features=self.num_classes,  # Number of output classes  输出类别数
        )(x)  # Fully connected layer  全连接层

        return y  # Output tensor  输出张量


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
    model = SimpleCNN(num_classes=num_classes)

    # Initialize parameters  初始化参数
    params = model.init(key, x)

    # Forward pass  前向传播
    logits = model.apply(params, x, train=False)

    print("SimpleCNN logits shape:", logits.shape)  # Predicted logits shape: (4, 5)  预测的 logits 形状：(4, 5)
