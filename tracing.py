#! python3

import torch
from model import Block, GPTConfig

x = torch.randn(2, 10, 768)  # [batch, seq_len, embed_dim]
print(f"Input shape: {x.shape}")
print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

block = Block(GPTConfig())

# After LayerNorm
x_norm = block.ln_1(x)
print(f"After LN1 mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")

# After attention + residual
x_attn = x + block.attn(x_norm)
print(f"After attention shape: {x_attn.shape}")
print(f"Residual magnitude: {(x_attn - x).abs().mean():.4f}")