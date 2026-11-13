from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RLNavMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        condition_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ELU()
        self.token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.condition_proj = build_mlp(int(condition_dim), hidden_dims, int(embed_dim), activation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.global_proj = build_mlp(int(embed_dim), hidden_dims, int(embed_dim), activation)
        self.output_dim = int(embed_dim) * 3

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.forward_with_attention(tokens, condition, need_weights=False)
        return encoded

    def forward_with_attention(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_features = self.token_proj(tokens)
        condition_feature = self.condition_proj(condition)
        query = condition_feature.unsqueeze(1)
        if need_weights:
            attended, weights = self.cross_attention(
                query,
                token_features,
                token_features,
                need_weights=True,
                average_attn_weights=bool(average_attn_weights),
            )
        else:
            attended, weights = self.cross_attention(query, token_features, token_features, need_weights=False)
        global_feature = self.global_proj(token_features.mean(dim=1))
        encoded = torch.cat([attended.squeeze(1), global_feature, condition_feature], dim=-1)
        return encoded, weights


class RLNavConditionedQueryMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        condition_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dims: Sequence[int] = (256,),
        crop_hidden_dims: Sequence[int] | None = None,
        query_hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ELU()
        if crop_hidden_dims is None:
            crop_hidden_dims = hidden_dims
        if query_hidden_dims is None:
            query_hidden_dims = hidden_dims

        self.token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.condition_proj = build_mlp(int(condition_dim), hidden_dims, int(embed_dim), activation)
        self.crop_proj = build_mlp(int(embed_dim), crop_hidden_dims, int(embed_dim), activation)
        self.query_proj = build_mlp(int(embed_dim) * 2, query_hidden_dims, int(embed_dim), activation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_dim = int(embed_dim) * 3

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.forward_with_attention(tokens, condition, need_weights=False)
        return encoded

    def forward_with_attention(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_features = self.token_proj(tokens)
        condition_feature = self.condition_proj(condition)
        crop_embedding = self.crop_proj(token_features.mean(dim=1))
        query_feature = self.query_proj(torch.cat([condition_feature, crop_embedding], dim=-1))
        query = query_feature.unsqueeze(1)
        if need_weights:
            attended, weights = self.cross_attention(
                query,
                token_features,
                token_features,
                need_weights=True,
                average_attn_weights=bool(average_attn_weights),
            )
        else:
            attended, weights = self.cross_attention(query, token_features, token_features, need_weights=False)
        encoded = torch.cat([attended.squeeze(1), crop_embedding, condition_feature], dim=-1)
        return encoded, weights


class RLNavDualScaleConditionedQueryMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        condition_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dims: Sequence[int] = (256,),
        local_crop_hidden_dims: Sequence[int] | None = None,
        global_hidden_dims: Sequence[int] | None = None,
        query_hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ELU()
        if local_crop_hidden_dims is None:
            local_crop_hidden_dims = hidden_dims
        if global_hidden_dims is None:
            global_hidden_dims = hidden_dims
        if query_hidden_dims is None:
            query_hidden_dims = hidden_dims

        self.local_token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.global_token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.condition_proj = build_mlp(int(condition_dim), hidden_dims, int(embed_dim), activation)
        self.local_crop_proj = build_mlp(int(embed_dim), local_crop_hidden_dims, int(embed_dim), activation)
        self.global_proj = build_mlp(int(embed_dim), global_hidden_dims, int(embed_dim), activation)
        self.query_proj = build_mlp(int(embed_dim) * 3, query_hidden_dims, int(embed_dim), activation)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_dim = int(embed_dim) * 4

    def forward(self, local_tokens: torch.Tensor, global_tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.forward_with_attention(local_tokens, global_tokens, condition, need_weights=False)
        return encoded

    def forward_with_attention(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        condition: torch.Tensor,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_features = self.local_token_proj(local_tokens)
        global_features = self.global_token_proj(global_tokens)
        condition_feature = self.condition_proj(condition)
        local_embedding = self.local_crop_proj(local_features.mean(dim=1))
        global_embedding = self.global_proj(global_features.mean(dim=1))
        query_feature = self.query_proj(torch.cat([condition_feature, local_embedding, global_embedding], dim=-1))
        query = query_feature.unsqueeze(1)
        if need_weights:
            attended, weights = self.cross_attention(
                query,
                local_features,
                local_features,
                need_weights=True,
                average_attn_weights=bool(average_attn_weights),
            )
        else:
            attended, weights = self.cross_attention(query, local_features, local_features, need_weights=False)
        encoded = torch.cat([attended.squeeze(1), local_embedding, global_embedding, condition_feature], dim=-1)
        return encoded, weights


class RLNavDualScalePooledMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        condition_dim: int,
        embed_dim: int = 128,
        hidden_dims: Sequence[int] = (256,),
        local_context_hidden_dims: Sequence[int] | None = None,
        local_crop_hidden_dims: Sequence[int] | None = None,
        global_hidden_dims: Sequence[int] | None = None,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ELU()
        if local_context_hidden_dims is None:
            local_context_hidden_dims = local_crop_hidden_dims
        if local_context_hidden_dims is None:
            local_context_hidden_dims = hidden_dims
        if local_crop_hidden_dims is None:
            local_crop_hidden_dims = hidden_dims
        if global_hidden_dims is None:
            global_hidden_dims = hidden_dims

        self.local_token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.global_token_proj = build_mlp(int(token_dim), hidden_dims, int(embed_dim), activation)
        self.condition_proj = build_mlp(int(condition_dim), hidden_dims, int(embed_dim), activation)
        self.local_context_proj = build_mlp(int(embed_dim), local_context_hidden_dims, int(embed_dim), activation)
        self.local_crop_proj = build_mlp(int(embed_dim), local_crop_hidden_dims, int(embed_dim), activation)
        self.global_proj = build_mlp(int(embed_dim), global_hidden_dims, int(embed_dim), activation)
        self.output_dim = int(embed_dim) * 4

    def forward(self, local_tokens: torch.Tensor, global_tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        local_features = self.local_token_proj(local_tokens)
        global_features = self.global_token_proj(global_tokens)
        condition_feature = self.condition_proj(condition)
        local_mean = local_features.mean(dim=1)
        local_context = self.local_context_proj(local_mean)
        local_embedding = self.local_crop_proj(local_mean)
        global_embedding = self.global_proj(global_features.mean(dim=1))
        return torch.cat([local_context, local_embedding, global_embedding, condition_feature], dim=-1)


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(_clone_activation(activation))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _clone_activation(activation: nn.Module) -> nn.Module:
    return activation.__class__()
