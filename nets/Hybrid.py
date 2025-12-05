import jax
import flax.linen as nn
from typing import Optional, Literal, Dict, Any

from nets.CNN import SimpleCNN
from nets.ResNet import ResNet18, ResNet34
from nets.Mamba import VisionMamba as Mamba  # 你现在 train.py 里就是这样 import 的

FusionType = Literal["concat_head", "weighted_sum", "gated_sum"]


class HybridMambaCNN(nn.Module):
    num_classes: int
    dropout_rate: float = 0.0

    # 选择 CNN 结构（调用你现成的 SimpleCNN）
    cnn_dropout_rate: Optional[float] = None

    # Mamba 配置（与 train.py 的 mamba_config 对齐）
    mamba_config: Optional[Dict[str, Any]] = None

    fusion: FusionType = "concat_head"
    fusion_hidden: int = 256  # concat_head / gated_sum 的中间维度

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 注意：不能在 __call__ 里修改 self.mamba_config，只能用局部变量
        if self.mamba_config is None:
            mamba_cfg = dict(
                patch_size=16,
                embed_dim=256,
                use_class_token=True,
                depth=4,
                conv_kernel_size=3,
                ssm_expend=2,
                ssm_d_state=8,
                ssm_dt_rank=8,
            )
        else:
            mamba_cfg = self.mamba_config

        # ---------- branch 1: CNN ----------
        cnn = SimpleCNN(
            num_classes=self.num_classes,
            dropout_rate=(
                self.cnn_dropout_rate
                if self.cnn_dropout_rate is not None
                else self.dropout_rate
            ),
        )
        logits_cnn = cnn(x, train=train)  # (B, num_classes)

        # ---------- branch 2: Mamba ----------
        mamba = Mamba(
            num_classes=self.num_classes,
            patch_size=mamba_cfg["patch_size"],
            embed_dim=mamba_cfg["embed_dim"],
            use_class_token=mamba_cfg["use_class_token"],
            depth=mamba_cfg["depth"],
            conv_kernel_size=mamba_cfg["conv_kernel_size"],
            ssm_expend=mamba_cfg["ssm_expend"],
            ssm_d_state=mamba_cfg["ssm_d_state"],
            ssm_dt_rank=mamba_cfg["ssm_dt_rank"],
            dropout_rate=self.dropout_rate,
        )
        logits_mamba = mamba(x, train=train)  # (B, num_classes)

        # ---------- fusion ----------
        if self.fusion == "weighted_sum":
            # learnable scalar alpha in [0,1]
            logit_alpha = self.param("logit_alpha", nn.initializers.zeros, (1,))
            alpha = jax.nn.sigmoid(logit_alpha)[0]
            logits = alpha * logits_mamba + (1.0 - alpha) * logits_cnn
            return logits

        elif self.fusion == "gated_sum":
            # sample-wise gating based on concatenated logits
            z = jax.numpy.concatenate([logits_mamba, logits_cnn], axis=-1)  # (B, 2C)
            h = nn.Dense(self.fusion_hidden)(z)
            h = nn.silu(h)
            if self.dropout_rate > 0 and train:
                h = nn.Dropout(rate=self.dropout_rate)(
                    h, deterministic=not train
                )
            gate = nn.Dense(1)(h)  # (B,1)
            alpha = jax.nn.sigmoid(gate)  # (B,1)
            logits = alpha * logits_mamba + (1 - alpha) * logits_cnn
            return logits

        else:  # "concat_head" 
            z = jax.numpy.concatenate([logits_mamba, logits_cnn], axis=-1)  # (B, 2C)
            h = nn.Dense(self.fusion_hidden)(z)
            h = nn.relu(h)
            if self.dropout_rate > 0 and train:
                h = nn.Dropout(rate=self.dropout_rate)(
                    h, deterministic=not train
                )
            logits = nn.Dense(self.num_classes)(h)
            return logits


class HybridMambaResNet(nn.Module):
    num_classes: int
    resnet_type: Literal["resnet18", "resnet34"] = "resnet18"
    dropout_rate: float = 0.0
    mamba_config: Optional[Dict[str, Any]] = None

    fusion: FusionType = "concat_head"
    fusion_hidden: int = 256

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 同样这里不能修改 self.mamba_config
        if self.mamba_config is None:
            mamba_cfg = dict(
                patch_size=16,
                embed_dim=256,
                use_class_token=True,
                depth=4,
                conv_kernel_size=3,
                ssm_expend=2,
                ssm_d_state=8,
                ssm_dt_rank=8,
            )
        else:
            mamba_cfg = self.mamba_config

        # ---------- branch 1: ResNet ----------
        if self.resnet_type == "resnet34":
            resnet = ResNet34(
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
            )
        else:
            resnet = ResNet18(
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
            )
        logits_resnet = resnet(x, train=train)

        # ---------- branch 2: Mamba ----------
        mamba = Mamba(
            num_classes=self.num_classes,
            patch_size=mamba_cfg["patch_size"],
            embed_dim=mamba_cfg["embed_dim"],
            use_class_token=mamba_cfg["use_class_token"],
            depth=mamba_cfg["depth"],
            conv_kernel_size=mamba_cfg["conv_kernel_size"],
            ssm_expend=mamba_cfg["ssm_expend"],
            ssm_d_state=mamba_cfg["ssm_d_state"],
            ssm_dt_rank=mamba_cfg["ssm_dt_rank"],
            dropout_rate=self.dropout_rate,
        )
        logits_mamba = mamba(x, train=train)

        # ---------- fusion ----------
        if self.fusion == "weighted_sum":
            logit_alpha = self.param("logit_alpha", nn.initializers.zeros, (1,))
            alpha = jax.nn.sigmoid(logit_alpha)[0]
            return alpha * logits_mamba + (1 - alpha) * logits_resnet

        elif self.fusion == "gated_sum":
            z = jax.numpy.concatenate([logits_mamba, logits_resnet], axis=-1)
            h = nn.Dense(self.fusion_hidden)(z)
            h = nn.silu(h)
            if self.dropout_rate > 0 and train:
                h = nn.Dropout(rate=self.dropout_rate)(
                    h, deterministic=not train
                )
            gate = nn.Dense(1)(h)
            alpha = jax.nn.sigmoid(gate)
            return alpha * logits_mamba + (1 - alpha) * logits_resnet

        else:
            z = jax.numpy.concatenate([logits_mamba, logits_resnet], axis=-1)
            h = nn.Dense(self.fusion_hidden)(z)
            h = nn.relu(h)
            if self.dropout_rate > 0 and train:
                h = nn.Dropout(rate=self.dropout_rate)(
                    h, deterministic=not train
                )
            return nn.Dense(self.num_classes)(h)


# ===== Comprehensive Hybrid Test Code =====
if __name__ == "__main__":
    import traceback
    import numpy as np
    import time
    import optax
    import jax
    import jax.numpy as jnp

    print("=" * 80)
    print("HYBRID (MAMBA + CNN/RESNET) COMPREHENSIVE TEST")
    print("=" * 80)


    # ------------------------------------------------------------
    # Util functions
    # ------------------------------------------------------------
    def count_params(params_dict):
        total = 0
        if isinstance(params_dict, dict):
            for v in params_dict.values():
                total += count_params(v)
        else:
            if hasattr(params_dict, "size"):
                total += params_dict.size
        return total


    def compute_grad_norm(g):
        if isinstance(g, dict):
            return jnp.sqrt(sum(compute_grad_norm(v) ** 2 for v in g.values()))
        else:
            return jnp.linalg.norm(g.flatten())


    def grad_stats(g):
        if isinstance(g, dict):
            all_vals = []
            for v in g.values():
                all_vals.extend(grad_stats(v))
            return all_vals
        else:
            return g.flatten().tolist()


    def print_param_tree(params_dict, depth=0):
        indent = "    " + "  " * depth
        for k, v in params_dict.items():
            if isinstance(v, dict):
                print(f"{indent}{k}:")
                print_param_tree(v, depth + 1)
            else:
                if hasattr(v, "shape"):
                    print(f"{indent}{k}: {v.shape}")


    def cross_entropy_loss(logits, labels_onehot):
        log_softmax = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(labels_onehot * log_softmax, axis=-1))
        return loss


    def accuracy(logits, labels_int):
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == labels_int)


    # ------------------------------------------------------------
    # ===== 1. Setup =====
    # ------------------------------------------------------------
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

    # dummy input
    key_init = jax.random.fold_in(key, 0)
    x_dummy = jax.random.normal(key_init, (batch_size, height, width, channels))

    # labels
    labels_int = jnp.array([0, 1, 2, 3, 0])[:batch_size]
    y_onehot = jax.nn.one_hot(labels_int, num_classes)

    # ------------------------------------------------------------
    # ===== Model variants to test =====
    # ------------------------------------------------------------
    print("\n[2] Preparing model variants...")

    model_variants = [
        ("HybridMambaCNN/concat_head",
         lambda: HybridMambaCNN(
             num_classes=num_classes,
             dropout_rate=0.2,
             fusion="concat_head",
             fusion_hidden=256,
         )),

        ("HybridMambaCNN/weighted_sum",
         lambda: HybridMambaCNN(
             num_classes=num_classes,
             dropout_rate=0.2,
             fusion="weighted_sum",
         )),

        ("HybridMambaCNN/gated_sum",
         lambda: HybridMambaCNN(
             num_classes=num_classes,
             dropout_rate=0.2,
             fusion="gated_sum",
             fusion_hidden=128,
         )),

        ("HybridMambaResNet18/concat_head",
         lambda: HybridMambaResNet(
             num_classes=num_classes,
             resnet_type="resnet18",
             dropout_rate=0.2,
             fusion="concat_head",
             fusion_hidden=256,
         )),

        ("HybridMambaResNet34/concat_head",
         lambda: HybridMambaResNet(
             num_classes=num_classes,
             resnet_type="resnet34",
             dropout_rate=0.2,
             fusion="concat_head",
             fusion_hidden=256,
         )),
    ]

    print("    ✓ Model variants ready:")
    for name, _ in model_variants:
        print(f"      - {name}")

    # ------------------------------------------------------------
    # Run comprehensive tests for each variant
    # ------------------------------------------------------------
    for variant_id, (variant_name, build_model) in enumerate(model_variants):
        print("\n" + "#" * 80)
        print(f"TESTING VARIANT [{variant_id + 1}/{len(model_variants)}]: {variant_name}")
        print("#" * 80)

        # --------------------------------------------------------
        # ===== 3. Create Model =====
        # --------------------------------------------------------
        print("\n[3] Creating model...")
        try:
            model = build_model()
            print("    ✓ Model created")
        except Exception as e:
            print(f"    ✗ Failed to create model: {e}")
            traceback.print_exc()
            continue

        # --------------------------------------------------------
        # ===== 4. Initialize Parameters =====
        # --------------------------------------------------------
        print("\n[4] Initializing model parameters...")
        try:
            key_model_init = jax.random.fold_in(key, 10 + variant_id)
            variables = model.init(
                {"params": key_model_init, "dropout": key_model_init},
                x_dummy,
                train=True,
            )
            params = variables["params"]
            batch_stats = variables.get("batch_stats", None)

            total_params = count_params(params)
            print("    ✓ Parameters initialized successfully")
            print(f"    Total parameters: {total_params:,}")

        except Exception as e:
            print(f"    ✗ Failed to initialize parameters: {e}")
            traceback.print_exc()
            continue

        # --------------------------------------------------------
        # ===== 5. Forward Pass Test (Training Mode) =====
        # --------------------------------------------------------
        print("\n[5] Testing forward pass (training mode)...")
        try:
            key_forward = jax.random.fold_in(key, 20 + variant_id)
            x_test = jax.random.normal(key_forward, (batch_size, height, width, channels))

            # apply with mutable batch_stats if exists
            if batch_stats is not None:
                logits, mutated = model.apply(
                    {"params": params, "batch_stats": batch_stats},
                    x_test,
                    train=True,
                    mutable=["batch_stats"],
                    rngs={"dropout": key_forward},
                )
                batch_stats_new = mutated["batch_stats"]
            else:
                logits = model.apply(
                    {"params": params},
                    x_test,
                    train=True,
                    rngs={"dropout": key_forward},
                )
                batch_stats_new = None

            print("    ✓ Forward pass successful")
            print(f"      Output shape: {logits.shape}")
            assert logits.shape == (batch_size, num_classes), \
                f"Shape mismatch! Got {logits.shape}, expected {(batch_size, num_classes)}"
            print("    ✓ Output shape verified")

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
            continue

        # --------------------------------------------------------
        # ===== 6. Forward Pass Test (Evaluation Mode) =====
        # --------------------------------------------------------
        print("\n[6] Testing forward pass (evaluation mode)...")
        try:
            if batch_stats is not None:
                logits_eval = model.apply(
                    {"params": params, "batch_stats": batch_stats_new},
                    x_test,
                    train=False,
                )
            else:
                logits_eval = model.apply(
                    {"params": params},
                    x_test,
                    train=False,
                )

            print("    ✓ Evaluation pass successful")
            print(f"      Output shape: {logits_eval.shape}")

            diff = jnp.abs(logits - logits_eval).max()
            print(f"      Max difference (train vs eval): {float(diff):.6f}")

            logits_eval_np = np.array(logits_eval)
            if np.isnan(logits_eval_np).any():
                print("    ✗ WARNING: Evaluation output contains NaN!")
            else:
                print("    ✓ Evaluation output is valid")

        except Exception as e:
            print(f"    ✗ Evaluation pass failed: {e}")
            traceback.print_exc()
            continue

        # --------------------------------------------------------
        # ===== 7. Loss Function Test =====
        # --------------------------------------------------------
        print("\n[7] Testing loss computation...")
        try:
            loss_val = cross_entropy_loss(logits_eval, y_onehot)
            print("    ✓ Loss computation successful")
            print(f"      Loss value: {float(loss_val):.4f}")

            if jnp.isnan(loss_val):
                print("    ✗ WARNING: Loss is NaN!")
            elif jnp.isinf(loss_val):
                print("    ✗ WARNING: Loss is Inf!")
            else:
                print("    ✓ Loss is valid")

        except Exception as e:
            print(f"    ✗ Loss computation failed: {e}")
            traceback.print_exc()
            continue

        # --------------------------------------------------------
        # ===== 8. Gradient Test =====
        # --------------------------------------------------------
        print("\n[8] Testing backward pass (gradient computation)...")
        try:
            def loss_fn_for_grad(p):
                if batch_stats is not None:
                    l = model.apply({"params": p, "batch_stats": batch_stats_new},
                                    x_test, train=False)
                else:
                    l = model.apply({"params": p}, x_test, train=False)
                return cross_entropy_loss(l, y_onehot)


            loss_value, grads = jax.value_and_grad(loss_fn_for_grad)(params)

            print("    ✓ Gradient computation successful")
            print(f"      Loss: {float(loss_value):.4f}")

            grad_norm = compute_grad_norm(grads)
            print(f"      Gradient norm: {float(grad_norm):.6f}")

            if jnp.isnan(grad_norm):
                print("    ✗ WARNING: Gradients contain NaN!")
            elif jnp.isinf(grad_norm):
                print("    ✗ WARNING: Gradients contain Inf!")
            else:
                print("    ✓ Gradients are valid")

            all_grads = jnp.array(grad_stats(grads))
            print(f"      Gradient min: {all_grads.min():.6f}")
            print(f"      Gradient max: {all_grads.max():.6f}")
            print(f"      Gradient mean: {all_grads.mean():.6f}")
            print(f"      Gradient std: {all_grads.std():.6f}")

        except Exception as e:
            print(f"    ✗ Gradient computation failed: {e}")
            traceback.print_exc()
            continue

        # --------------------------------------------------------
        # ===== 9. Mini Training Loop Test =====
        # --------------------------------------------------------
        print("\n[9] Testing mini training loop (5 steps)...")
        try:
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=1e-3)
            )
            opt_state = optimizer.init(params)


            def loss_fn_train(p, x_local, y_local):
                if batch_stats is not None:
                    logits_local = model.apply({"params": p, "batch_stats": batch_stats_new},
                                               x_local, train=False)
                else:
                    logits_local = model.apply({"params": p},
                                               x_local, train=False)
                return cross_entropy_loss(logits_local, y_local)


            @jax.jit
            def train_step(p, opt_s, x_local, y_local):
                loss_val, grads = jax.value_and_grad(
                    lambda pp: loss_fn_train(pp, x_local, y_local)
                )(p)
                updates, opt_s = optimizer.update(grads, opt_s)
                p = jax.tree_util.tree_map(lambda a, u: a + u, p, updates)
                return p, opt_s, loss_val


            losses = []
            print("    Running 5 training steps...")
            p_local, opt_s_local = params, opt_state

            for step in range(5):
                key_step = jax.random.fold_in(key, 1000 + step + variant_id)
                x_batch = jax.random.normal(key_step, (batch_size, height, width, channels))
                p_local, opt_s_local, loss_step = train_step(p_local, opt_s_local, x_batch, y_onehot)

                losses.append(float(loss_step))
                print(f"      Step {step + 1}: loss = {losses[-1]:.6f}")

                if jnp.isnan(loss_step):
                    print(f"    ✗ NaN detected at step {step + 1}!")
                    break

            print("    ✓ Training loop completed")
            print(f"      Loss trajectory: {[f'{l:.6f}' for l in losses]}")

            if len(losses) > 1:
                loss_change = losses[0] - losses[-1]
                print(f"      Loss change: {loss_change:.6f}")
                if loss_change > 0:
                    print("    ✓ Loss is decreasing (good sign!)")

        except Exception as e:
            print(f"    ✗ Training loop failed: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # ===== 10. Inference Speed Test =====
        # --------------------------------------------------------
        print("\n[10] Testing inference speed...")
        try:
            @jax.jit
            def infer_jit(p, bs, x_local):
                if bs is not None:
                    return model.apply({"params": p, "batch_stats": bs},
                                       x_local, train=False)
                else:
                    return model.apply({"params": p},
                                       x_local, train=False)


            # warm up
            _ = infer_jit(params, batch_stats_new, x_test)

            num_iterations = 20
            start = time.time()
            for _ in range(num_iterations):
                _ = infer_jit(params, batch_stats_new, x_test)
            elapsed = time.time() - start

            time_per_batch = elapsed / num_iterations * 1000
            throughput = batch_size * num_iterations / (elapsed + 1e-8)

            print("    ✓ Inference speed test completed")
            print(f"      Time per batch: {time_per_batch:.2f} ms")
            print(f"      Throughput: {throughput:.1f} samples/sec")

        except Exception as e:
            print(f"    ✗ Speed test failed: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # ===== 11. Different Batch Sizes Test =====
        # --------------------------------------------------------
        print("\n[11] Testing with different batch sizes...")
        try:
            for test_batch in [1, 2, 4, 8, 16]:
                x_test_batch = jax.random.normal(
                    jax.random.fold_in(key, 2000 + test_batch + variant_id),
                    (test_batch, height, width, channels)
                )
                if batch_stats is not None:
                    logits_batch = model.apply(
                        {"params": params, "batch_stats": batch_stats_new},
                        x_test_batch,
                        train=False
                    )
                else:
                    logits_batch = model.apply(
                        {"params": params},
                        x_test_batch,
                        train=False
                    )
                print(f"    ✓ Batch size {test_batch:2d}: output shape {logits_batch.shape}")

        except Exception as e:
            print(f"    ✗ Batch size test failed: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # ===== 12. Architecture / Param Shape =====
        # --------------------------------------------------------
        print("\n[12] Architecture analysis...")
        try:
            print(f"    Model architecture: {variant_name}")
            print(f"    Total parameters : {total_params:,}")
            print("    ✓ Architecture analysis completed")
        except Exception as e:
            print(f"    ✗ Architecture analysis failed: {e}")
            traceback.print_exc()

        print("\n[13] Verifying parameter shapes...")
        try:
            print_param_tree(params)
        except Exception as e:
            print(f"    ✗ Parameter verification failed: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # ===== 14. Activation Output Analysis =====
        # --------------------------------------------------------
        print("\n[14] Analyzing activation outputs (logits stats)...")
        try:
            logits_np = np.array(logits_eval)
            print("    ✓ Activation analysis completed")
            print(f"      Logits - Min: {logits_np.min():.6f}, Max: {logits_np.max():.6f}")
            print(f"      Logits - Mean: {logits_np.mean():.6f}, Std: {logits_np.std():.6f}")
        except Exception as e:
            print(f"    ✗ Activation analysis failed: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # Summary for this variant
        # --------------------------------------------------------
        print("\n" + "-" * 80)
        print(f"SUMMARY for {variant_name}")
        print(f"✓ Model forward/backward/training sanity checks passed")
        print(f"✓ Model has {total_params:,} trainable parameters")
        print("-" * 80)

    # ------------------------------------------------------------
    # Global Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ALL HYBRID VARIANTS TESTED")
    print("=" * 80)
