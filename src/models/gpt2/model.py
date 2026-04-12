from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPT2Config:
    """Defaults match GPT-2 small."""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-5



class GPT2Attention(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


class GPT2MLP(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_ratio * config.hidden_size, bias=True)
        self.fc2 = nn.Linear(config.mlp_ratio * config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.wte(input_ids) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class GPT2ForCausalLM(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [B, S, vocab_size]."""
        return self.lm_head(self.model(input_ids))
