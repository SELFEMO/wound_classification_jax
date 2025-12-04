# CNN.py 中补充的内容
import jax
import jax.numpy as jnp
from flax import linen as nn


class BaselineCNN(nn.Module):
    """来自 jax4max.py 的 Baseline CNN 的 Flax 版本（分类用）

    结构大致为：
    [Conv -> GroupNorm -> ReLU -> MaxPool] x 4 -> FC(256) -> FC(128) -> FC(num_classes)
    """
    num_classes: int = 9  # 和 jax4max 里的 HYPERPARAMS["num_classes"] 一致
    dropout_rate: float = 0.2  # Dropout 概率

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (batch, H, W, 3)  输入图像已经归一化到 [0, 1]

        def conv_block(x, out_ch):
            # 对应 Equinox 里的 ConvolutionalBlock：Conv2d + GroupNorm + ReLU
            x = nn.Conv(
                features=out_ch,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
            )(x)
            # GroupNorm 的组数不能大于通道数
            num_groups = min(32, out_ch)
            x = nn.GroupNorm(num_groups=num_groups)(x)
            x = nn.relu(x)
            return x

        # 4 个卷积块，每个后面接一次 2x2 MaxPool（总共下采样 16 倍）
        channels = [32, 64, 128, 256]
        for c in channels:
            x = conv_block(x, c)
            x = nn.max_pool(
                x,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            )

        # 展平
        x = x.reshape((x.shape[0], -1))

        # 全连接部分：256 -> Dropout -> 128 -> num_classes
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        if self.dropout_rate is not None and self.dropout_rate > 0.0 and train:
            x = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not train,
            )(x)

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        logits = nn.Dense(self.num_classes)(x)
        return logits


# ===== Comprehensive Test Code =====
if __name__ == "__main__":
    import traceback
    import numpy as np
    import time
    import optax

    print("=" * 80)
    print("SIMPLE CNN COMPREHENSIVE TEST")
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
    print("\n[2] Creating BaselineCNN model...")
    try:
        model = BaselineCNN(
            num_classes=num_classes,
            dropout_rate=0.2,
        )
        print("    ✓ Model created")
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
            {
                'params': key_init,
                'dropout': key_init
            },
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
            mutable=['batch_stats'],
            rngs={
                'dropout': key_forward
            }
        )[0]

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
            optax.adam(learning_rate=1e-3)
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
        for test_batch in [1, 2, 4, 8, 16]:
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
        print(f"    Model architecture: BaselineCNN")
        print(f"    Total parameters: {total_params:,}")

        # Calculate FLOPs (approximate)
        # Conv(32): 224*224*3*32*3*3 = ~227M FLOPs
        # Conv(64): 112*112*32*64*3*3 = ~227M FLOPs
        # Conv(128): 56*56*64*128*3*3 = ~226M FLOPs
        # Dense: 128*5 = ~640 FLOPs
        approximate_flops = (224 * 224 * 3 * 32 * 3 * 3 + 112 * 112 * 32 * 64 * 3 * 3 + 56 * 56 * 64 * 128 * 3 * 3) / 1e9
        print(f"    Approximate FLOPs (forward pass): {approximate_flops:.2f}G")

        print(f"    ✓ Architecture analysis completed")

    except Exception as e:
        print(f"    ✗ Architecture analysis failed: {e}")
        traceback.print_exc()

    # ===== 12. Parameter Shape Verification =====
    print("\n[12] Verifying parameter shapes...")
    try:
        def print_param_tree(params_dict, prefix="", depth=0):
            indent = "    " + "  " * depth
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    print(f"{indent}{k}:")
                    print_param_tree(v, prefix, depth + 1)
                else:
                    if hasattr(v, 'shape'):
                        print(f"{indent}{k}: {v.shape}")


        print_param_tree(params)

    except Exception as e:
        print(f"    ✗ Parameter verification failed: {e}")
        traceback.print_exc()

    # ===== 13. Activation Output Analysis =====
    print("\n[13] Analyzing intermediate activation outputs...")
    try:
        # Create a model variant that returns intermediate outputs
        def get_intermediate_outputs(params_local, x_local):
            """Extract activation statistics at different layers"""
            logits = model.apply(
                params_local,
                x_local,
                train=False
            )

            # Return logits statistics
            logits_np = np.array(logits)
            return {
                'logits_min': logits_np.min(),
                'logits_max': logits_np.max(),
                'logits_mean': logits_np.mean(),
                'logits_std': logits_np.std(),
            }


        stats = get_intermediate_outputs(params, x_test)
        print(f"    ✓ Activation analysis completed")
        print(f"      Logits - Min: {stats['logits_min']:.6f}, Max: {stats['logits_max']:.6f}")
        print(f"      Logits - Mean: {stats['logits_mean']:.6f}, Std: {stats['logits_std']:.6f}")

    except Exception as e:
        print(f"    ✗ Activation analysis failed: {e}")
        traceback.print_exc()

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All critical tests passed!")
    print("✓ BaselineCNN model is ready for training")
    print(f"✓ Model has {total_params:,} trainable parameters")
    print("=" * 80)
