import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from core.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, ParallelLMHead
from models.gpt2.model import GPT2Config, GPT2ForCausalLM, GPT2Attention, GPT2MLP


def _parallelize_attention(attn: GPT2Attention, tp_mesh: DeviceMesh) -> None:
    attn.q_proj = ColumnParallelLinear(attn.q_proj.in_features, attn.q_proj.out_features, tp_mesh)
    attn.k_proj = ColumnParallelLinear(attn.k_proj.in_features, attn.k_proj.out_features, tp_mesh)
    attn.v_proj = ColumnParallelLinear(attn.v_proj.in_features, attn.v_proj.out_features, tp_mesh)
    attn.o_proj = RowParallelLinear(attn.o_proj.in_features, attn.o_proj.out_features, tp_mesh)
    attn.num_heads //= tp_mesh.size()


def _parallelize_mlp(mlp: GPT2MLP, tp_mesh: DeviceMesh) -> None:
    mlp.fc1 = ColumnParallelLinear(mlp.fc1.in_features, mlp.fc1.out_features, tp_mesh)
    mlp.fc2 = RowParallelLinear(mlp.fc2.in_features, mlp.fc2.out_features, tp_mesh)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (VocabParallelEmbedding, ParallelLMHead, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(config: GPT2Config, tp_mesh: DeviceMesh, device: str = "cuda") -> GPT2ForCausalLM:
    """Build a parallelized GPT-2 without ever holding full weights on one device."""
    with torch.device("meta"):
        model = GPT2ForCausalLM(config)
    parallelize(model, tp_mesh)
    model.to_empty(device=device)
    model.apply(_init_weights)
    return model


def parallelize(model: GPT2ForCausalLM, tp_mesh: DeviceMesh) -> GPT2ForCausalLM:
    wte = model.model.wte
    model.model.wte = VocabParallelEmbedding(wte.num_embeddings, wte.embedding_dim, tp_mesh)

    for block in model.model.blocks:
        _parallelize_attention(block.attn, tp_mesh)
        _parallelize_mlp(block.mlp, tp_mesh)

    lm_head = model.lm_head
    model.lm_head = ParallelLMHead(lm_head.out_features, lm_head.in_features, tp_mesh)

    return model
