import torch
from torch import nn
import torch.nn.functional as F
import random, math, sys
# from .util import mask_

def mask_(matrices: torch.Tensor, maskval: float, mask_diagonal: bool) -> torch.Tensor:
    # matrices shape: [..., T, T]
    T = matrices.size(-1)
    # diagonal=0  → include diagonal; diagonal=1 → exclude diagonal
    diag = 0 if mask_diagonal else 1
    # Upper-triangular mask of shape [T, T]
    tri = torch.triu(torch.ones((T, T), dtype=torch.bool, device=matrices.device), diagonal=diag)
    # Broadcast over leading dims and fill in-place
    return matrices.masked_fill_(tri, maskval)

# def mask_(matrices, maskval=0.0, mask_diagonal=True):
#     """
#     Masks out all values in the given batch of matrices where i <= j holds,
#     i < j if mask_diagonal is false

#     In place operation

#     :param tns:
#     :return:
#     """

#     h, w = matrices.size(-2), matrices.size(-1)

#     indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
#     matrices[..., indices[0], indices[1]] = maskval

class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=8, mask=False, kqnorm=False, scalefactor=None):
        """

        :param emb: The dimension of the input and output vectors.
        :param heads: The number of heads (parallel executions of the self-attention)
        :param mask: Whether to apply an autoregressive mask.
        :param kqnorm: Whether to apply layer normalization to the keys and queries.
        :param scalefactor: Multiplier for the attention weights. If none, the default `1/sqrt(emb/heads)` is used,
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

        self.scalefactor = 1/math.sqrt(emb // heads) if scalefactor is None else scalefactor

    def forward(self, x):

        b, t, e = x.size() # [batch size, sequence length, embedding dim]
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)


        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries
        keys    = keys

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot * self.scalefactor

        assert dot.size() == (b*h, t, t)

        # if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
        # mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        mask_(dot, maskval=float(-1e9), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # -- dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default',
                 pos_embedding=None, sa_kwargs={}):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask, **sa_kwargs)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(), #nn.ReLU(), # GELU: GPT2 uses gelu here
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    
class RCTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(
            self, 
            emb, 
            heads, 
            depth, 
            seq_length, 
            obs_act_dim, 
            output_dim, 
            causal=False,
            dropout=0.0,             
        ):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()
        # Create sinusoidal positional embeddings
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2) * (-math.log(10000.0) / emb))
        pos_embedding = torch.zeros(seq_length, emb)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)

        self.input_projection = nn.Linear(obs_act_dim, emb)
        self.output_projection = nn.Linear(emb, output_dim)

        self.embed_ln = nn.LayerNorm(emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=causal, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.do = nn.Dropout(dropout)

    def forward(self, obs_act_history):
        """
        :param obs_act_history: Tensor of shape (batch, seq_len, obs_act_dim)
        :return: Predicted output vector from final linear projection.

        """
        obs_act_history = obs_act_history.contiguous()
        embeddings = self.input_projection(obs_act_history)
        b, t, e = embeddings.size()
        # cosine positional embeddings
        positions = self.pos_embedding[:t, :].to(embeddings.device).unsqueeze(0).expand(b, t, e)

        x = embeddings + positions
        # x = self.embed_ln(x)
        x = self.do(x)
        x = self.tblocks(x)
        # 1. Last token representation (like BERT [CLS] token)
        latent = x[:, -1, :]  # [batch, emb]
    
        return  latent # self.output_projection(latent) # latent #
