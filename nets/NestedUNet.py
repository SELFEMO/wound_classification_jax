import jax
import jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn
from typing import Sequence, List, Tuple


class DoubleConv(nn.Module):  # 定义一个双卷积块
    """
    (convolution => [BN] => ReLU) * 2
    """
    inChannels: int
    middleChannels: int
    outChannels: int
    preBatchNorm: bool = False  # 是否使用 BN+ReLU+Conv 的顺序

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        :param x: 输入数据 (N, H, W, C)
        :param train: 是否训练模式
        :return: 双卷积块的结果
        """
        axis = -1  # 通道在最后一维

        if self.preBatchNorm:
            # BN + ReLU + Conv 的顺序
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                axis=axis,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.middleChannels,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
            )(x)

            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                axis=axis,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.outChannels,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
            )(x)
        else:
            # Conv + BN + ReLU 的顺序
            x = nn.Conv(
                features=self.middleChannels,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
            )(x)
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                axis=axis,
            )(x)
            x = nn.relu(x)

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
                axis=axis,
            )(x)
            x = nn.relu(x)

        return x  # 返回双卷积块的结果


class Backbone(nn.Module):  # 定义一个UNet++的主干网络
    """
    Down-sampling => Down-sampling => Down-sampling => Down-sampling
    """
    inChannels: int
    filters: Sequence[int] = (64, 128, 256, 512, 1024)

    def setup(self):
        # 输入块，包含一个双卷积块
        self.inc = DoubleConv(self.inChannels, self.filters[0], self.filters[0])
        # 下采样部分，包含四个下采样块
        self.down1 = DoubleConv(self.filters[0], self.filters[1], self.filters[1])
        self.down2 = DoubleConv(self.filters[1], self.filters[2], self.filters[2])
        self.down3 = DoubleConv(self.filters[2], self.filters[3], self.filters[3])
        self.down4 = DoubleConv(self.filters[3], self.filters[4], self.filters[4])

    @nn.compact
    def __call__(self, x, train: bool = True) -> List[jnp.ndarray]:
        """
        :param x: 输入数据 (N, H, W, C)
        :return: 主干网络（输入块和四次下采样部分）的结果
        """
        # 最大池化操作
        def max_pool(z):
            return nn.max_pool(z, window_shape=(2, 2), strides=(2, 2), padding="VALID")

        # 输入部分
        x1 = self.inc(x, train=train)
        # 下采样部分
        x2 = self.down1(max_pool(x1), train=train)
        x3 = self.down2(max_pool(x2), train=train)
        x4 = self.down3(max_pool(x3), train=train)
        x5 = self.down4(max_pool(x4), train=train)
        return [x1, x2, x3, x4, x5]


class Up(nn.Module):  # 定义一个上采样块
    """
    上采样 + 拼接 + DoubleConv
    """
    inChannels: int
    middleChannels: int
    outChannels: int

    def _upsample_to(self, x, ref):
        """
        使用双线性插值将 x 上采样到 ref 相同的空间尺寸
        """
        n, h, w, c = ref.shape
        return jimage.resize(x, (n, h, w, c), method="bilinear")

    @nn.compact
    def __call__(self, x_list: List[jnp.ndarray], train: bool = True):
        """
        :param x_list: 输入数据列表 (同一层上的多个特征图)
        :return: 上采样块的结果
        """
        # 上采样操作：把最后一个特征图上采样到与第一个特征图同尺寸
        x_last = x_list[-1]
        x_ref = x_list[0]
        x_last_up = self._upsample_to(x_last, x_ref)

        x_list_new = list(x_list)
        x_list_new[-1] = x_last_up

        # 将 x_list 中的所有张量在通道维进行拼接
        x = jnp.concatenate(x_list_new, axis=-1)

        # 双卷积块
        x = DoubleConv(self.inChannels, self.middleChannels, self.outChannels)(
            x, train=train
        )
        return x  # 返回上采样块的结果


class OutConv(nn.Module):  # 定义一个输出块
    """
    Conv2d => (可选激活)
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
        return x  # 返回输出块的结果


class NestedUNet(nn.Module):  # 定义一个UNet++类
    """
    UNet++类，包含深监督输出
    """
    inChannels: int
    outChannels: int
    filters: Sequence[int] = (64, 128, 256, 512, 1024)

    def setup(self):
        # 主干网络，包含一个输入块和四个下采样块
        self.backbone = Backbone(self.inChannels, filters=self.filters)
        # 第一层上采样块
        self.up01 = Up(self.filters[0] + self.filters[1], self.filters[0], self.filters[0])
        self.up11 = Up(self.filters[1] + self.filters[2], self.filters[1], self.filters[1])
        self.up21 = Up(self.filters[2] + self.filters[3], self.filters[2], self.filters[2])
        self.up31 = Up(self.filters[3] + self.filters[4], self.filters[3], self.filters[3])
        # 第二层上采样块
        self.up02 = Up(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.up12 = Up(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.up22 = Up(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])
        # 第三层上采样块
        self.up03 = Up(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.up13 = Up(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])
        # 第四层上采样块
        self.up04 = Up(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])
        # 输出块
        self.outConv = OutConv(self.filters[0], self.outChannels)

    def __call__(self, x, train: bool = True) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        :param x: 输入数据 (N, H, W, C)
        :param train: 是否训练模式
        :return: 主输出和 4 个深监督输出
        """
        # 主干网络（输入块和四次下采样部分）
        x00, x10, x20, x30, x40 = self.backbone(x, train=train)

        # 第一层上采样块
        x01 = self.up01([x00, x10], train=train)
        x11 = self.up11([x10, x20], train=train)
        x21 = self.up21([x20, x30], train=train)
        x31 = self.up31([x30, x40], train=train)

        # 第二层上采样块
        x02 = self.up02([x00, x01, x11], train=train)
        x12 = self.up12([x10, x11, x21], train=train)
        x22 = self.up22([x20, x21, x31], train=train)

        # 第三层上采样块
        x03 = self.up03([x00, x01, x02, x12], train=train)
        x13 = self.up13([x10, x11, x12, x22], train=train)

        # 第四层上采样块
        x04 = self.up04([x00, x01, x02, x03, x13], train=train)

        # 输出部分：主输出 + 4 个深监督输出
        main_out = self.outConv(x04)
        aux_outs = [
            self.outConv(x01),
            self.outConv(x02),
            self.outConv(x03),
            self.outConv(x04),
        ]
        return main_out, aux_outs


# ========== 测试代码 (JAX 版本) ==========
if __name__ == "__main__":
    # ---------------- 基本配置 ----------------
    in_channels = 3      # 根据你的任务修改
    out_channels = 1     # 根据你的任务修改
    batch_size = 2
    height = 256         # 建议为 16 的倍数，适配 4 次下采样 / 上采样
    width = 256

    # JAX 自动选择设备
    device = jax.devices()[0]
    print(f"Using device: {device}")

    # ---------------- 构建模型 ----------------
    model = NestedUNet(inChannels=in_channels, outChannels=out_channels)
    print(model)

    # ---------------- 构造假数据 (NHWC) ----------------
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, height, width, in_channels))
    print(f"\nInput shape: {x.shape}")

    # ---------------- 初始化参数 ----------------
    variables = model.init(key, x, train=False)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # 统计可训练参数数量
    def count_parameters(pytree):
        leaves = jax.tree_util.tree_leaves(pytree)
        return int(sum([p.size for p in leaves]))

    total_params = count_parameters(params)
    print(f"\nTrainable parameters: {total_params}")

    # ---------------- 前向传播（推理模式） ----------------
    main_out, aux_outs = model.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        train=False,
    )
    print(f"\nMain output shape: {main_out.shape}")
    assert main_out.shape == (batch_size, height, width, out_channels), \
        f"Main output shape is wrong: {main_out.shape}"

    assert isinstance(aux_outs, (list, tuple)) and len(aux_outs) == 4, \
        f"Expected 4 auxiliary outputs, got {len(aux_outs)}"

    for i, aux in enumerate(aux_outs):
        print(f"Aux[{i}] shape: {aux.shape}")
        assert aux.shape == (batch_size, height, width, out_channels), \
            f"Aux[{i}] shape is wrong: {aux.shape}"

    # ---------------- 反向传播测试 ----------------
    def loss_fn(params, batch_stats, x_in):
        variables = {"params": params, "batch_stats": batch_stats}
        (main_y, aux_ys), new_vars = model.apply(
            variables,
            x_in,
            train=True,
            mutable=["batch_stats"],
        )
        loss_main = jnp.mean(main_y)
        if isinstance(aux_ys, (list, tuple)) and len(aux_ys) > 0:
            loss_aux = sum(jnp.mean(a) for a in aux_ys) / len(aux_ys)
            loss = loss_main + loss_aux
        else:
            loss = loss_main
        new_batch_stats = new_vars["batch_stats"]
        return loss, new_batch_stats

    x_train = jax.random.normal(jax.random.PRNGKey(1), (1, height, width, in_channels))

    (loss_value, new_batch_stats), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, batch_stats, x_train)

    print(f"\nDummy loss (main + aux mean): {float(loss_value):.6f}")

    # 简单检查梯度是否正常
    grad_leaves = jax.tree_util.tree_leaves(grads)
    max_grad = max(
        [float(jnp.abs(g).max()) for g in grad_leaves if g is not None]
    )
    print(f"Backward pass OK, max grad: {max_grad:.6e}")

    print("\nAll tests finished. If no assertion error or exception occurred, the NestedUNet JAX/Flax implementation can now perform forward and backward computations normally on the GPU.")
