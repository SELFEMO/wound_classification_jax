import jax
import flax

class MambaBlock(flax.linen.Module):
    """
    A single Mamba block.  单个 Mamba 块。
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

        A = init_matrix('A', (self.d_state, self.d_state))  # State matrix 状态矩阵
        Bm = init_matrix('Bm', (self.d_state, self.d_state))  # Input matrix 输入矩阵
        C = init_matrix('C', (self.d_model, self.d_state))  # Output matrix 输出矩阵
        D = init_matrix('D', (self.d_model, self.d_state))  # Feedthrough matrix 直通矩阵
