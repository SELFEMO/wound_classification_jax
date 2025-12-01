import jax
import flax


class MambaBlock(flax.linen.Module):
    """
    A single Mamba block.  单个 Mamba 块。

    Mamba is a state space model that combines linear state space dynamics with gating mechanisms.
    Mamba 是一种状态空间模型，结合了线性状态空间动力学和门控机制。
    """
    d_model: int
    d_state: int

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the Mamba block.

        :param x: Input tensor 输入张量
        :param train: Whether the model is in training mode 模型是否处于训练模式

        :return: Output tensor after passing through the Mamba block  通过 Mamba 块后的输出张量
        """
        # Input: (B, T, d_model)
        B, T, d_model = x.shape

        # Linear Layer 线性层
        in_projection = flax.linen.Dense(
            features=self.d_state,  # Output dimension 输出维度
            use_bias=True,  # Use bias 使用偏置
        )(x)  # Shape: (B, T, d_state)
        gate_dense = flax.linen.Dense(
            features=self.d_model,  # Output dimension 输出维度
            use_bias=True,  # Use bias 使用偏置
        )
        out_projection = flax.linen.Dense(
            features=self.d_model,  # Output dimension 输出维度
            use_bias=True,  # Use bias 使用偏置
        )

        # Full sequence one-time projection  全序列一次投影
        u = in_projection  # Shape: (B, T, d_state)
        gate_logits = gate_dense(x)  # Shape: (B, T, d_model)
        gate_vals = flax.linen.sigmoid(gate_logits)  # Shape: (B, T, d_model)

        # State space matrix parameters  状态空间矩阵参数
        def init_matrix(
                name: str,
                shape: tuple
        ):
            """
            Initializes a state space matrix.

            :param name: Name of the matrix  矩阵的名称
            :param shape: Shape of the matrix  矩阵的形状
            :return: Initialized matrix  初始化矩阵
            """
            return self.param(
                name,  # Parameter name  参数名称
                flax.linen.initializers.lecun_normal(),  # Initializer 初始化器
                shape,  # Shape of the matrix  矩阵的形状
            )

        A = init_matrix(
            name='A',
            shape=(self.d_state, self.d_state)
        )  # State matrix 状态矩阵
        Bm = init_matrix(
            name='Bm',
            shape=(self.d_state, self.d_state)
        )  # Input matrix 输入矩阵
        C = init_matrix(
            name='C',
            shape=(self.d_model, self.d_state)
        )  # Output matrix 输出矩阵
        D = init_matrix(
            name='D',
            shape=(self.d_model, self.d_state)
        )  # Feedthrough matrix 直通矩阵

        # Scan needs time in dimension 0  scan 需要 time 在第 0 维  Scan needs time in dimension 0
        u_seq = jax.numpy.swapaxes(u, 0, 1)  # Shape: (T, B, d_state)
        g_seq = jax.numpy.swapaxes(gate_vals, 0, 1)  # Shape: (T, B, d_model)
        x_seq = jax.numpy.swapaxes(x, 0, 1)  # Shape: (T, B, d_model)

        def mamba_step(
                h_prev: jax.numpy.ndarray,
                inputs: tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]
        ) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
            """
            Single step of the Mamba state space model.

            :param h_prev: Previous hidden state 上一个隐藏状态
            :param inputs: Tuple of input at current time step and gate value 当前时间步的输入和门值的元组

            :return: New hidden state and output at current time step 新的隐藏状态和当前时间步的输出
            """
            # Unpack inputs 解包输入
            u_t, g_t, x_t = inputs  # u_t: (B, d_state), g_t: (B, d_model)

            # h_t = A @ h_prev + Bm @ u_t
            h_t = jax.numpy.einsum("bd,dd->bd", h_prev, A) + jax.numpy.einsum("bd,dd->bd", u_t, Bm)
            h_t = jax.nn.tanh(h_t)

            # y_t = C @ h_t + D @ u_t
            y_t = jax.numpy.einsum("bd,md->bm", h_t, C) + jax.numpy.einsum("bd,md->bm", u_t, D)

            # Apply gate: gate + residual connection  应用门：门 + 残差连接
            y_t = g_t * y_t + (1 - g_t) * x_t  # Shape: (B, d_model)

            return h_t, y_t  # Return new hidden state and output 返回新的隐藏状态和输出

        # Initialize hidden state 初始化隐藏状态
        h0 = jax.numpy.zeros((B, self.d_state))  # Shape: (B, d_state)
        # Scan over the time dimension 在时间维度上扫描
        _, ys = jax.lax.scan(
            mamba_step,  # Function to scan 扫描的函数
            h0,  # Initial hidden state 初始隐藏状态
            xs=(
                u_seq, g_seq, x_seq
            )  # Inputs to the scan 扫描的输入
        )  # y_sequence: (T, B, d_model)
        ys = jax.numpy.swapaxes(ys, 0, 1)  # Shape: (B, T, d_model)

        # Final output projection 最终输出投影
        output = out_projection(ys)  # Shape: (B, T, d_model)
        output = output + x  # Residual connection 残差连接

        return output  # Shape: (B, T, d_model)


class MambaClassifier(flax.linen.Module):
    """
    Mamba-based classifier. 基于 Mamba 的分类器。
    """
    d_model: int
    d_state: int
    num_classes: int
    num_blocks: int = 2

    @flax.linen.compact
    def __call__(
            self,
            x_seq: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the Mamba classifier.

        :param x: Input tensor 输入张量
        :param train: Whether the model is in training mode 模型是否处于训练模式

        :return: Logits for each class 每个类的 logits
        """
        x = x_seq  # Shape: (B, T, d_model)

        # Pass through Mamba blocks  通过 Mamba 块
        for i in range(self.num_blocks):
            x = MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
            )(x, train=train)  # Shape: (B, T, d_model)

        # Pooling: Mean over time dimension 池化：时间维度的均值
        x = jax.numpy.mean(x, axis=1)  # Shape: (B, d_model)

        # Final classification layer 最终分类层
        logits = flax.linen.Dense(
            features=self.num_classes,  # Number of classes 类别数
            use_bias=True,  # Use bias 使用偏置
        )(x)  # Shape: (B, num_classes)

        return logits  # Shape: (B, num_classes)


# ===== Test Code =====
if __name__ == "__main__":
    # Select device  选择设备
    device = jax.devices()[0]
    print(f"Using device: {device}")

    # Input parameters  输入参数
    key = jax.random.PRNGKey(0)
    batch_size = 4
    T = 64
    d_model = 128
    d_state = 64
    num_classes = 5

    # Random input tensor  随机输入张量
    x_seq = jax.random.normal(key, (batch_size, T, d_model))

    # Build model  构建模型
    model = MambaClassifier(
        num_classes=num_classes,
        d_model=d_model,
        d_state=d_state,
        num_blocks=2,
    )

    # Initialize parameters  初始化参数
    params = model.init(key, x_seq)

    # Forward pass  前向传播
    logits = model.apply(params, x_seq, train=False)

    print("MambaClassifier logits shape:", logits.shape)  # 预期: (4, 5)
