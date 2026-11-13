import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

import math
import torch
import torch.nn as nn

# input x has shape [B, T, C] (Batch, Tokens, Channels)
class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)  # Learnable scaling factor
        self.gamma = nn.Parameter(torch.ones(C))  # Scale parameter (similar to LN)
        self.beta = nn.Parameter(torch.zeros(C))  # Shift parameter

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
    
class RGTransformer(nn.Module):
    """
    Causal Transformer for robotics.
    """

    def __init__(
            self, 
            emb, 
            heads, 
            depth, 
            seq_length, 
            obs_act_dim, 
            output_dim, 
            input_hidden_dims=[512, 256, 128], 
            output_hidden_dims=[512, 256, 128], 
            input_output_activation=nn.ReLU,
            attention_type='default',
        ):

        super().__init__()

        self.obs_act_dim = obs_act_dim

        # Create sinusoidal positional embeddings
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2) * (-math.log(10000.0) / emb))
        pos_embedding = torch.zeros(seq_length, emb)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)

        # Linear projection layer for concatenated obs+action          
        # Create input projection MLP
        input_projection_layers = []
        input_dims = [obs_act_dim] + input_hidden_dims
        for i in range(len(input_dims)-1):
            input_projection_layers.append(nn.Linear(input_dims[i], input_dims[i+1]))
            input_projection_layers.append(input_output_activation())
        # Add final layer to match embedding dimension
        input_projection_layers.append(nn.Linear(input_dims[-1], emb))
        self.input_projection = nn.Sequential(*input_projection_layers)

        # project embeding to output dim
        output_projection_layers = []
        output_dims = [emb] + output_hidden_dims
        for i in range(len(output_dims)-1):
            output_projection_layers.append(nn.Linear(output_dims[i], output_dims[i+1]))
            output_projection_layers.append(input_output_activation())
        # Add final layer to match output dim
        output_projection_layers.append(nn.Linear(output_dims[-1], output_dim))
        self.output_projection = nn.Sequential(*output_projection_layers)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb, 
                    heads=heads, 
                    seq_length=seq_length, 
                    mask=True, 
                    attention_type=attention_type,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)


    def forward(self, obs_act_history):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # Add validation
        if obs_act_history.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {obs_act_history.dim()}D")
        if obs_act_history.size(2) != self.obs_act_dim:
            raise ValueError(f"Expected {self.obs_act_dim} features, got {obs_act_history.size(2)}")

        batch_size, seq_length, obs_act_dim = obs_act_history.size()

        # Project to embedding dimension
        embeddings = self.input_projection(obs_act_history)
        b, t, e = embeddings.size()

        # Add positional embeddings
        positions = self.pos_embedding[:t, :].to(embeddings.device).unsqueeze(0).expand(b, t, e)
        x = embeddings + positions

        x = self.tblocks(x)

        # Extract last timestep's output
        last_timestep = x[:, -1, :]  # [batch, emb]

        # Continuous output prediction
        return self.output_projection(last_timestep)
    
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
            # num_tokens, 
            # num_classes, 

            obs_act_dim, 
            output_dim, 
            input_hidden_dims=[512, 256, 128], 
            output_hidden_dims=[512, 256, 128], 
            input_output_activation=nn.ReLU,

            max_pool=True, 
            dropout=0.0, 
            wide=False,
            
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

        # self.max_pool =  max_pool

        # self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        # self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        # Create sinusoidal positional embeddings
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2) * (-math.log(10000.0) / emb))
        pos_embedding = torch.zeros(seq_length, emb)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)
        # print("pos_embedding device:", self.pos_embedding.device)


        # Linear projection layer for concatenated obs+action          
        # # Create input projection MLP
        # input_projection_layers = []
        # input_dims = [obs_act_dim] + input_hidden_dims
        # for i in range(len(input_dims)-1):
        #     input_projection_layers.append(nn.Linear(input_dims[i], input_dims[i+1]))
        #     input_projection_layers.append(input_output_activation())
        # # Add final layer to match embedding dimension
        # input_projection_layers.append(nn.Linear(input_dims[-1], emb))
        # self.input_projection = nn.Sequential(*input_projection_layers)
        
        # ###
        self.input_projection = nn.Linear(obs_act_dim, emb)
        # ###
        
        # # project embeding to output dim
        # output_projection_layers = []
        # output_dims = [emb] + output_hidden_dims
        # for i in range(len(output_dims)-1):
        #     output_projection_layers.append(nn.Linear(output_dims[i], output_dims[i+1]))
        #     output_projection_layers.append(input_output_activation())
        # # Add final layer to match output dim
        # output_projection_layers.append(nn.Linear(output_dims[-1], output_dim))
        # self.output_projection = nn.Sequential(*output_projection_layers)

        # ###
        
        self.output_projection = nn.Linear(emb, output_dim)
        # ###

        self.embed_ln = nn.LayerNorm(emb)
        # self.embed_norm_dtanh = DyT(emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        # self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)
        # print(f"transformer emb: {emb}, transformer output_dim: {output_dim}")


    def forward(self, obs_act_history):
        """
        :param obs_act_history: Tensor of shape (batch, seq_len, obs_act_dim)
        :return: Predicted output vector from final linear projection.

        """
        # tokens = self.token_embedding(x)
        # Project to embedding dimension
        # print(f"obs_act_history device: {obs_act_history.device}")
        device = next(self.parameters()).device
        # # Ensure input is on correct device
        # if obs_act_history.device != device:
        #     obs_act_history = obs_act_history.to(device)

        if torch.isnan(obs_act_history).any() or torch.isinf(obs_act_history).any():
            raise ValueError("obs_act_history contains NaNs or Infs.")

        assert obs_act_history.ndim == 3
        assert obs_act_history.dtype == torch.float32
        assert not torch.isnan(obs_act_history).any()
        assert not torch.isinf(obs_act_history).any()
        obs_act_history = obs_act_history.contiguous()


        embeddings = self.input_projection(obs_act_history)
        b, t, e = embeddings.size()

        # b, t, e = tokens.size()

        # cosine positional embeddings
        positions = self.pos_embedding[:t, :].to(device).unsqueeze(0).expand(b, t, e)

        # print("pos_embedding device:", positions.device)
        # Create position indices tensor
        # positions_idx = torch.arange(t, device=embeddings.device)
        # # Get position embeddings and expand to match batch size
        # positions = self.pos_embedding(positions_idx).unsqueeze(0).expand(b, t, e)

        x = embeddings + positions

        x = self.embed_ln(x)

        # x = self.embed_norm_dtanh(x)

        x = self.do(x)

        x = self.tblocks(x)

        # x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        # x = self.toprobs(x)

        # 1. Last token representation (like BERT [CLS] token)
        latent = x[:, -1, :]  # [batch, emb]
        
        # 2. Max pooling across sequence
        # latent = x.max(dim=1)[0]  # [batch, emb]
        
        # 3. Average pooling across sequence
        # latent = x.mean(dim=1)  # [batch, emb]
        
        # 4. Concatenate multiple pooling methods
        # latent = torch.cat([
        #     x[:, -1, :],         # last token
        #     x.max(dim=1)[0],     # max pooling
        #     x.mean(dim=1)        # average pooling
        # ], dim=-1)  # [batch, 3*emb]

        return self.output_projection(latent)
    
class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=(seq_length * 2 - 1 if attention_type=='relative' else seq_length))


        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type, pos_embedding=self.pos_embedding))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
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

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

