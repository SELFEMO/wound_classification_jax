import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class DoubleConv(nn.Module):  # 定义一个双卷积块
    """
    (convolution => [BN] => ReLU) * 2
    """
    inChannels: int
    outChannels: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        :param x: 输入数据 (N, H, W, C)
        :param train: 是否训练模式（影响 BN）
        :return: 双卷积块的结果
        """
        # 第一次卷积操作：卷积 + BN + ReLU
        x = nn.Conv(
            features=self.outChannels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis=-1,
        )(x)
        x = nn.relu(x)

        # 第二次卷积操作：卷积 + BN + ReLU
        x = nn.Conv(
            features=self.outChannels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            axis=-1,
        )(x)
        x = nn.relu(x)

        return x  # 返回双卷积块的结果


class Down(nn.Module):  # 定义一个下采样块
    """
    Maxpool => DoubleConv
    """
    inChannels: int
    outChannels: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        :param x: 输入数据
        :param train: 是否训练模式
        :return: 下采样块的结果
        """
        # 下采样操作，包含最大池化和双卷积操作
        # 最大池化
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        # 双卷积块
        x = DoubleConv(self.inChannels, self.outChannels)(x, train=train)
        return x  # 返回下采样块的结果


class Up(nn.Module):  # 定义一个上采样块
    """
    UpConv (ConvTranspose2d => [BN] => ReLU) * 2 => Conv2d
    """
    inChannels: int
    outChannels: int

    @nn.compact
    def __call__(self, x1, x2, train: bool = True):
        """
        :param x1: 输入数据1（来自下方尺度）
        :param x2: 输入数据2（来自跳跃连接）
        :param train: 是否训练模式
        :return: 上采样块的结果
        """
        # 上采样操作，使用反卷积（转置卷积）
        x1_ = nn.ConvTranspose(
            features=self.inChannels // 2,  # 输出通道数
            kernel_size=(2, 2),  # 卷积核大小
            strides=(2, 2),  # 步长
            padding="SAME",
            use_bias=False,
        )(x1)

        # 将 x1_ 和 x2 进行拼接（通道维在最后一维）
        x = jnp.concatenate([x2, x1_], axis=-1)

        # 双卷积块
        x = DoubleConv(self.inChannels, self.outChannels)(x, train=train)
        return x  # 返回上采样块的结果


class OutConv(nn.Module):  # 定义一个输出块
    """
    Conv2d => (可选 Sigmoid)
    """
    inChannels: int
    outChannels: int

    @nn.compact
    def __call__(self, x):
        """
        :param x: 输入数据
        :return: 输出块的结果
        """
        x = nn.Conv(
            features=self.outChannels,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
        )(x)
        # 如需 sigmoid / softmax，可在外部自行加激活
        return x  # 返回输出块的结果


class UNet(nn.Module):
    """
    UNet类 (JAX + Flax 版本)
    """
    inChannels: int
    outChannels: int
    # 定义五个下采样块的通道数
    filters: Sequence[int] = (64, 128, 256, 512, 1024)

    def setup(self):
        # 输入块，包含一个双卷积块
        self.inc = DoubleConv(self.inChannels, self.filters[0])
        # 下采样块，包含四个下采样块
        self.down1 = Down(self.filters[0], self.filters[1])
        self.down2 = Down(self.filters[1], self.filters[2])
        self.down3 = Down(self.filters[2], self.filters[3])
        self.down4 = Down(self.filters[3], self.filters[4])
        # 上采样块，包含四个上采样块
        self.up1 = Up(self.filters[4], self.filters[3])
        self.up2 = Up(self.filters[3], self.filters[2])
        self.up3 = Up(self.filters[2], self.filters[1])
        self.up4 = Up(self.filters[1], self.filters[0])
        # 输出块，包含一个输出块
        self.outc = OutConv(self.filters[0], self.outChannels)

    def __call__(self, x, train: bool = True):
        """
        :param x: 输入数据 (N, H, W, C)
        :param train: 是否训练模式
        :return: UNet的输出结果 y
        """
        # 输入块
        x1 = self.inc(x, train=train)
        # 下采样块
        x2 = self.down1(x1, train=train)
        x3 = self.down2(x2, train=train)
        x4 = self.down3(x3, train=train)
        x5 = self.down4(x4, train=train)
        # 上采样块
        x6 = self.up1(x5, x4, train=train)
        x7 = self.up2(x6, x3, train=train)
        x8 = self.up3(x7, x2, train=train)
        x9 = self.up4(x8, x1, train=train)
        # 输出块
        y = self.outc(x9)
        return y, None  # JAX 风格直接返回 y


# ========== 测试代码 (JAX 版本) ==========
if __name__ == "__main__":
    # ---------------- 基本配置 ----------------
    in_channels = 3  # 输入通道数，可按你的任务修改
    out_channels = 1  # 输出通道数，可按你的任务修改
    batch_size = 2
    height = 256  # 高和宽建议为 16 的倍数，方便 4 次下采样 / 上采样
    width = 256

    # JAX 会自动选择设备
    device = jax.devices()[0]
    print(f"Using device: {device}")

    # ---------------- 构建模型 ----------------
    model = UNet(inChannels=in_channels, outChannels=out_channels)
    print(model)

    # ---------------- 构造假数据 (NHWC) ----------------
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, height, width, in_channels))
    print(f"\nInput shape: {x.shape}")

    # ---------------- 初始化参数 ----------------
    # 含 BatchNorm，会得到 'params' 和 'batch_stats' 两个 collection
    variables = model.init(key, x, train=False)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # 统计可训练参数数量
    def count_parameters(pytree):
        leaves = jax.tree_util.tree_leaves(pytree)
        return int(sum([p.size for p in leaves]))

    total_params = count_parameters(params)
    print(f"\nTrainable parameters: {total_params}")

    # ---------------- 前向传播 ----------------
    main_out, _ = model.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        train=False,
    )
    print(f"Output shape: {main_out.shape}")

    # 形状检查
    assert main_out.shape == (batch_size, height, width, out_channels), \
        f"Output shape is wrong: {main_out.shape}"

    # ---------------- 反向传播测试 ----------------
    # 用一条样本做一次简单的 loss 和梯度，确认梯度能正常传播
    def loss_fn(params, batch_stats, x_in):
        """
        :param params: 模型参数
        :param batch_stats: BN 的统计量
        :param x_in: 输入
        :return: (loss, new_batch_stats)
        """
        variables = {"params": params, "batch_stats": batch_stats}
        # mutable=["batch_stats"] 时，apply 返回 (输出, 更新后的变量字典)
        (y_pred, _), new_variables = model.apply(
            variables,
            x_in,
            train=True,
            mutable=["batch_stats"],
        )
        loss = jnp.mean(y_pred)
        new_batch_stats = new_variables["batch_stats"]
        return loss, new_batch_stats

    x_train = jax.random.normal(jax.random.PRNGKey(1), (1, height, width, in_channels))

    # 这里 value_and_grad 的 has_aux=True，对应 loss_fn 的第二个返回值 new_batch_stats
    (loss_value, new_batch_stats), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, batch_stats, x_train)

    print(f"\nDummy loss (mean of output): {float(loss_value):.6f}")

    # 简单检查一下所有参数的梯度是否存在且非零
    grad_leaves = jax.tree_util.tree_leaves(grads)
    max_grad = max(
        [float(jnp.abs(g).max()) for g in grad_leaves if g is not None]
    )
    print(f"Backward pass OK, max grad: {max_grad:.6e}")

    print("\nAll tests finished. If no assertion error or exception occurred, the JAX/Flax UNet structure is working correctly.")
