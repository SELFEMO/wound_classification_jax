import jax
import flax

class MambaBlock(flax.linen.Module):
    """
    A single Mamba block.  单个 Mamba 块。

    Dense -> BatchNorm -> ReLU ->
    Dense -> BatchNorm -> Add(skip connection) -> ReLU
    """

