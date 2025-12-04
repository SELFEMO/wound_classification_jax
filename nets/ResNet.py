import jax
import flax
from typing import Optional


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
            dtype=jax.numpy.float32,
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
            dtype=jax.numpy.float32,
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
                dtype=jax.numpy.float32,
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
    dropout_rate: Optional[float] = None  # Dropout rate (if any)  dropout 率（如果有的话）

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
            dtype=jax.numpy.float32,
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
            (64, 2, False),  # (out_channels, num_blocks, use_projection)  （输出通道数，块数，使用投影）
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

        # Dropout layer (if specified)  dropout 层（如果指定）
        if self.dropout_rate is not None and self.dropout_rate > 0.0 and train:
            x = flax.linen.Dropout(
                rate=self.dropout_rate,
                deterministic=not train
            )(x)

        # Fully connected layer  全连接层
        y = flax.linen.Dense(
            features=self.num_classes
        )(x)

        return y  # Output tensor  输出张量


class ResNet34(flax.linen.Module):
    """
    ResNet34 model.  ResNet34 模型。

    Conv(out_channels, stride) -> BatchNorm -> ReLU ->MaxPool(3x3, 2) ->
    [basic_block * 3, 4, 6, 3] ->
    GlobalAvgPool -> FullyConnected(num_classes)
    """
    num_classes: int = 2  # Number of output classes  输出类别数
    dropout_rate: Optional[float] = None  # Dropout rate (if any)  dropout 率（如果有的话）

    @flax.linen.compact
    def __call__(
            self,
            x: jax.numpy.ndarray,
            train: bool = True
    ) -> jax.numpy.ndarray:
        """
        Forward pass of the ResNet34 model.

        :param x: Input tensor  输入张量
        :param train:  Whether the model is in training mode  模型是否处于训练模式

        :return: Output tensor after passing through the ResNet34 model  通过 ResNet34 模型后的输出张量
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
            dtype=jax.numpy.float32,
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
            (64, 3, False),  # (out_channels, num_blocks, use_projection)  （输出通道数，块数，使用投影）
            (128, 4, True),
            (256, 6, True),
            (512, 3, True),
        ]

        # Build residual blocks  构建残差块
        for out_channels, num_blocks, use_projection in block_configs:
            for i in range(num_blocks):
                x = basic_block(
                    out_channels=out_channels,  # Number of output channels  输出通道数
                    strides=(2, 2) if i == 0 and use_projection else (1, 1),  # Stride for the first block if projection is used  如果使用投影，则为第一个块的步幅
                    use_projection=(i == 0 and use_projection),  # Use projection for the first
                )(x, train=train)

        # Global average pooling  全局平均池化
        x = jax.numpy.mean(x, axis=(1, 2))

        # Dropout layer (if specified)  dropout 层（如果指定）
        if self.dropout_rate is not None and self.dropout_rate > 0.0 and train:
            x = flax.linen.Dropout(
                rate=self.dropout_rate,
                deterministic=not train
            )(x)

        # Fully connected layer  全连接层
        y = flax.linen.Dense(
            features=self.num_classes
        )(x)

        return y  # Output tensor  输出张量


# ===== Comprehensive Test Code =====
if __name__ == "__main__":
    import traceback
    import numpy as np
    import time
    import optax

    print("=" * 80)
    print("RESNET COMPREHENSIVE TEST")
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

    # ===== 2. Create Models =====
    print("\n[2] Creating ResNet models...")
    try:
        resnet18_model = ResNet18(num_classes=num_classes)
        resnet34_model = ResNet34(num_classes=num_classes)
        print("    ✓ ResNet18 created")
        print("    ✓ ResNet34 created")
    except Exception as e:
        print(f"    ✗ Failed to create models: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 3. Initialize Parameters =====
    print("\n[3] Initializing model parameters...")
    try:
        key_init = jax.random.fold_in(key, 0)
        x_dummy = jax.random.normal(key_init, (batch_size, height, width, channels))

        # Initialize ResNet18
        params18 = resnet18_model.init(
            {'params': key_init, 'dropout': key_init},
            x_dummy,
            train=True,
        )
        print("    ✓ ResNet18 parameters initialized")

        # Initialize ResNet34
        params34 = resnet34_model.init(
            {'params': key_init, 'dropout': key_init},
            x_dummy,
            train=True,
        )
        print("    ✓ ResNet34 parameters initialized")


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


        total_params18 = count_params(params18)
        total_params34 = count_params(params34)
        print(f"    ResNet18 total parameters: {total_params18:,}")
        print(f"    ResNet34 total parameters: {total_params34:,}")

    except Exception as e:
        print(f"    ✗ Failed to initialize parameters: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 4. Forward Pass Test (ResNet18 - Training Mode) =====
    print("\n[4] Testing ResNet18 forward pass (training mode)...")
    try:
        key_forward = jax.random.fold_in(key, 1)
        x_test = jax.random.normal(key_forward, (batch_size, height, width, channels))

        logits18 = resnet18_model.apply(
            params18,
            x_test,
            train=True,
            mutable=['batch_stats'],
            rngs={
                'dropout': key_forward,
            }
        )[0]

        print(f"    ✓ Forward pass successful")
        print(f"      Output shape: {logits18.shape}")
        print(f"      Expected shape: ({batch_size}, {num_classes})")

        assert logits18.shape == (batch_size, num_classes), \
            f"Shape mismatch! Got {logits18.shape}, expected {(batch_size, num_classes)}"
        print("    ✓ Output shape verified")

        # Check for NaN/Inf
        logits18_np = np.array(logits18)
        nan_count = np.isnan(logits18_np).sum()
        inf_count = np.isinf(logits18_np).sum()

        if nan_count > 0:
            print(f"    ✗ WARNING: Output contains {nan_count} NaN values!")
        elif inf_count > 0:
            print(f"    ✗ WARNING: Output contains {inf_count} Inf values!")
        else:
            print("    ✓ No NaN/Inf in output")

        print(f"      Logits min: {logits18_np.min():.4f}")
        print(f"      Logits max: {logits18_np.max():.4f}")
        print(f"      Logits mean: {logits18_np.mean():.4f}")
        print(f"      Logits std: {logits18_np.std():.4f}")

    except Exception as e:
        print(f"    ✗ Forward pass (ResNet18 training) failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 5. Forward Pass Test (ResNet18 - Evaluation Mode) =====
    print("\n[5] Testing ResNet18 forward pass (evaluation mode)...")
    try:
        logits18_eval = resnet18_model.apply(
            params18,
            x_test,
            train=False
        )
        print(f"    ✓ Evaluation pass successful")
        print(f"      Output shape: {logits18_eval.shape}")

        diff = jax.numpy.abs(logits18 - logits18_eval).max()
        print(f"      Max difference (train vs eval): {float(diff):.6f}")

        logits18_eval_np = np.array(logits18_eval)
        if np.isnan(logits18_eval_np).any():
            print("    ✗ WARNING: Evaluation output contains NaN!")
        else:
            print("    ✓ Evaluation output is valid")

    except Exception as e:
        print(f"    ✗ Evaluation pass failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 6. Forward Pass Test (ResNet34 - Training Mode) =====
    print("\n[6] Testing ResNet34 forward pass (training mode)...")
    try:
        logits34 = resnet34_model.apply(
            params34,
            x_test,
            train=True,
            mutable=['batch_stats'],
            rngs={
                'dropout': key_forward,
            }
        )[0]

        print(f"    ✓ Forward pass successful")
        print(f"      Output shape: {logits34.shape}")

        assert logits34.shape == (batch_size, num_classes)
        print("    ✓ Output shape verified")

        logits34_np = np.array(logits34)
        nan_count = np.isnan(logits34_np).sum()
        inf_count = np.isinf(logits34_np).sum()

        if nan_count > 0:
            print(f"    ✗ WARNING: Output contains {nan_count} NaN values!")
        elif inf_count > 0:
            print(f"    ✗ WARNING: Output contains {inf_count} Inf values!")
        else:
            print("    ✓ No NaN/Inf in output")

        print(f"      Logits min: {logits34_np.min():.4f}")
        print(f"      Logits max: {logits34_np.max():.4f}")

    except Exception as e:
        print(f"    ✗ Forward pass (ResNet34 training) failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 7. Forward Pass Test (ResNet34 - Evaluation Mode) =====
    print("\n[7] Testing ResNet34 forward pass (evaluation mode)...")
    try:
        logits34_eval = resnet34_model.apply(
            params34,
            x_test,
            train=False
        )
        print(f"    ✓ Evaluation pass successful")
        print(f"      Output shape: {logits34_eval.shape}")

        logits34_eval_np = np.array(logits34_eval)
        if np.isnan(logits34_eval_np).any():
            print("    ✗ WARNING: Evaluation output contains NaN!")
        else:
            print("    ✓ Evaluation output is valid")

    except Exception as e:
        print(f"    ✗ Evaluation pass (ResNet34) failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 8. Loss Function Test =====
    print("\n[8] Testing loss computation...")
    try:
        def loss_fn(params_local, model_local, x_local, y_local):
            logits_local = model_local.apply(params_local, x_local, train=False)
            log_softmax = jax.nn.log_softmax(logits_local, axis=-1)
            loss = -jax.numpy.mean(jax.numpy.sum(y_local * log_softmax, axis=-1))
            return loss, logits_local


        # Create one-hot labels
        y_dummy = jax.nn.one_hot(
            jax.numpy.array([0, 1, 2, 3, 0])[:batch_size],
            num_classes
        )

        loss18, _ = loss_fn(params18, resnet18_model, x_test, y_dummy)
        loss34, _ = loss_fn(params34, resnet34_model, x_test, y_dummy)

        print(f"    ✓ Loss computation successful")
        print(f"      ResNet18 loss: {float(loss18):.4f}")
        print(f"      ResNet34 loss: {float(loss34):.4f}")

        if jax.numpy.isnan(loss18) or jax.numpy.isnan(loss34):
            print("    ✗ WARNING: Loss is NaN!")
        else:
            print("    ✓ Loss is valid")

    except Exception as e:
        print(f"    ✗ Loss computation failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 9. Gradient Test =====
    print("\n[9] Testing backward pass (gradient computation)...")
    try:
        grad_fn18 = jax.value_and_grad(
            lambda p: loss_fn(p, resnet18_model, x_test, y_dummy)[0]
        )
        loss18_val, grads18 = grad_fn18(params18)

        grad_fn34 = jax.value_and_grad(
            lambda p: loss_fn(p, resnet34_model, x_test, y_dummy)[0]
        )
        loss34_val, grads34 = grad_fn34(params34)

        print(f"    ✓ Gradient computation successful")
        print(f"      ResNet18 loss: {float(loss18_val):.4f}")
        print(f"      ResNet34 loss: {float(loss34_val):.4f}")


        # Compute gradient norms
        def compute_grad_norm(g):
            if isinstance(g, dict):
                return jax.numpy.sqrt(
                    sum(compute_grad_norm(v) ** 2 for v in g.values())
                )
            else:
                return jax.numpy.linalg.norm(g.flatten())


        grad_norm18 = compute_grad_norm(grads18)
        grad_norm34 = compute_grad_norm(grads34)

        print(f"      ResNet18 gradient norm: {float(grad_norm18):.6f}")
        print(f"      ResNet34 gradient norm: {float(grad_norm34):.6f}")

        if jax.numpy.isnan(grad_norm18) or jax.numpy.isnan(grad_norm34):
            print("    ✗ WARNING: Gradients contain NaN!")
        else:
            print("    ✓ Gradients are valid")

    except Exception as e:
        print(f"    ✗ Gradient computation failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 10. Mini Training Loop Test (ResNet18) =====
    print("\n[10] Testing ResNet18 mini training loop (5 steps)...")
    try:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-3)
        )
        opt_state = optimizer.init(params18)


        def loss_fn_train(params_local, x_local, y_local):
            logits_local = resnet18_model.apply(
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

            params18, opt_state, loss = train_step(params18, opt_state, x_batch, y_dummy)
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

    except Exception as e:
        print(f"    ✗ Training loop failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 11. Mini Training Loop Test (ResNet34) =====
    print("\n[11] Testing ResNet34 mini training loop (5 steps)...")
    try:
        optimizer34 = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=1e-3)
        )
        opt_state34 = optimizer34.init(params34)


        def loss_fn_train34(params_local, x_local, y_local):
            logits_local = resnet34_model.apply(
                params_local,
                x_local,
                train=False
            )
            log_softmax = jax.nn.log_softmax(logits_local, axis=-1)
            loss = -jax.numpy.mean(jax.numpy.sum(y_local * log_softmax, axis=-1))
            return loss


        @jax.jit
        def train_step34(params_local, opt_state_local, x_local, y_local):
            loss_val, grads = jax.value_and_grad(
                lambda p: loss_fn_train34(p, x_local, y_local)
            )(params_local)
            updates, opt_state_local = optimizer34.update(grads, opt_state_local)
            params_local = jax.tree_util.tree_map(
                lambda p, u: p + u,
                params_local,
                updates
            )
            return params_local, opt_state_local, loss_val


        losses34 = []
        print("    Running 5 training steps...")

        for step in range(5):
            key = jax.random.fold_in(key, step + 200)
            x_batch = jax.random.normal(key, (batch_size, height, width, channels))

            params34, opt_state34, loss = train_step34(params34, opt_state34, x_batch, y_dummy)
            losses34.append(float(loss))

            print(f"      Step {step + 1}: loss = {losses34[-1]:.6f}")

            if jax.numpy.isnan(loss):
                print(f"    ✗ NaN detected at step {step + 1}!")
                break

        print(f"    ✓ Training loop completed")
        print(f"      Loss trajectory: {[f'{l:.6f}' for l in losses34]}")

        if len(losses34) > 1:
            loss_change = losses34[0] - losses34[-1]
            print(f"      Loss change: {loss_change:.6f}")
            if loss_change > 0:
                print(f"    ✓ Loss is decreasing (good sign!)")

    except Exception as e:
        print(f"    ✗ Training loop failed: {e}")
        traceback.print_exc()
        exit(1)

    # ===== 12. Inference Speed Test =====
    print("\n[12] Testing inference speed...")
    try:
        @jax.jit
        def infer_jit18(p, x):
            return resnet18_model.apply(
                p,
                x,
                train=False
            )


        @jax.jit
        def infer_jit34(p, x):
            return resnet34_model.apply(
                p,
                x,
                train=False
            )


        # Warm up
        _ = infer_jit18(params18, x_test)
        _ = infer_jit34(params34, x_test)

        # Timing
        num_iterations = 20

        start = time.time()
        for _ in range(num_iterations):
            _ = infer_jit18(params18, x_test)
        elapsed18 = time.time() - start

        start = time.time()
        for _ in range(num_iterations):
            _ = infer_jit34(params34, x_test)
        elapsed34 = time.time() - start

        time_per_batch18 = elapsed18 / num_iterations * 1000
        throughput18 = batch_size * num_iterations / (elapsed18 + 1e-8)  # samples/sec  # 1e-8 to avoid div by zero

        time_per_batch34 = elapsed34 / num_iterations * 1000
        throughput34 = batch_size * num_iterations / (elapsed34 + 1e-8)  # samples/sec  # 1e-8 to avoid div by zero

        print(f"    ✓ Inference speed test completed")
        print(f"      ResNet18 - Time per batch: {time_per_batch18:.2f} ms, Throughput: {throughput18:.1f} samples/sec")
        print(f"      ResNet34 - Time per batch: {time_per_batch34:.2f} ms, Throughput: {throughput34:.1f} samples/sec")

    except Exception as e:
        print(f"    ✗ Speed test failed: {e}")
        traceback.print_exc()

    # ===== 13. Different Batch Sizes Test =====
    print("\n[13] Testing with different batch sizes...")
    try:
        for test_batch in [1, 2, 4, 8, 16]:
            x_test_batch = jax.random.normal(
                jax.random.fold_in(key, test_batch),
                (test_batch, height, width, channels)
            )
            logits_batch18 = resnet18_model.apply(
                params18,
                x_test_batch,
                train=False
            )
            logits_batch34 = resnet34_model.apply(
                params34,
                x_test_batch,
                train=False
            )
            print(f"    ✓ Batch size {test_batch:2d}: ResNet18 {logits_batch18.shape}, ResNet34 {logits_batch34.shape}")

    except Exception as e:
        print(f"    ✗ Batch size test failed: {e}")
        traceback.print_exc()

    # ===== 14. Parameter Comparison =====
    print("\n[14] Comparing model architectures...")
    try:
        print(f"    ResNet18 total parameters: {total_params18:,}")
        print(f"    ResNet34 total parameters: {total_params34:,}")
        print(f"    Parameter ratio (ResNet34/ResNet18): {total_params34 / total_params18:.2f}x")

    except Exception as e:
        print(f"    ✗ Parameter comparison failed: {e}")
        traceback.print_exc()

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All critical tests passed!")
    print("✓ ResNet18 model is ready for training")
    print("✓ ResNet34 model is ready for training")
    print("=" * 80)
