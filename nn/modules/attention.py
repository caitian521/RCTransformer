import torch.nn as nn
import torch
from torch.nn.functional import softmax
from torch.nn.init import kaiming_normal_
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_q_k):
        """Scaled Dot-Product Attention model: :math:`softmax(QK^T/sqrt(dim))V`.

        Args:
            dim_q_k (int): dimension of `queries` and `keys`.

        Inputs: query, key, value, mask
            - **value** of shape `(batch, seq_len, dim_v)`:  a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_q_k)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_q_k)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`, default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs: attention, attention_weights
            - **attention** of shape `(batch, q_len, dim_v)` a float tensor containing attention
              along `query` and `value` with the corresponding `key`.
            - **attention_weights** of shape `(batch, q_len, seq_len)`: a float tensor containing distribution of
              attention weights.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = np.power(dim_q_k, -0.5)

    def forward(self, value, key, query, mask=None):
        # (batch, q_len, seq_len)
        adjacency = query.bmm(key.transpose(1, 2)) * self.scale_factor

        if mask is not None:
            adjacency.data.masked_fill_(mask.data, -float('inf'))

        attention = softmax(adjacency, 2)
        return attention.bmm(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout=0.1, lexical=False):
        #dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout
        #n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout
        """Multi-Head Attention model.

        Args:
            n_heads (int): number of heads.
            dim_m (int): hidden size of model.
            dim_q_k (int): dimension of projection `queries` and `keys`.
            dim_v (int): dimension of projection `values`.
            dropout (float, optional): dropout probability.

        Inputs:
            - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`: default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs:
            - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
              along `query` and `value` with the corresponding `key` using Multi-Head Attention mechanism.
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_m = dim_m
        self.dim_q_k = dim_q_k
        self.dim_v = dim_v

        self.query_projection = nn.Parameter(torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.key_projection = nn.Parameter(torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.value_projection = nn.Parameter(torch.FloatTensor(n_heads, dim_m, dim_v))

        self.attention = ScaledDotProductAttention(dim_q_k)
        self.output = nn.Linear(dim_v * n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

        # Initialize projection tensors
        for parameter in [self.query_projection, self.key_projection, self.value_projection]:
            kaiming_normal_(parameter.data)

        if lexical:
            self.ek_normal = nn.Linear(dim_m, dim_m)
            self.hk_normal = nn.Linear(dim_m, dim_m)
            self.ev_normal = nn.Linear(dim_m, dim_m)
            self.hv_normal = nn.Linear(dim_m, dim_m)
            self.gate_k = nn.Linear(dim_m,dim_m)
            self.gate_v = nn.Linear(dim_m,dim_m)



    def forward(self, value, key, query, mask=None, embeddings=None):
        if embeddings is not None:
            key, value = self.lexical_shortcut(embeddings, key, value)

        seq_len = key.shape[1]
        q_len = query.shape[1]
        batch_size = query.shape[0]

        residual = query
        # (batch, x, dim_m) -> (n_heads, batch * x, dim_m)

        value, key, query = map(self.stack_heads, [value, key, query])

        if mask is not None:
            mask = self.stack_mask(mask)

        # (n_heads, batch * x, dim_m) -> (n_heads, batch * x, projection) -> (n_heads * batch, x, projection)
        # where `projection` is `dim_q_k`, `dim_v` for each input respectively.
        # value_projection  (n_heads, dim_m, dim_q_k)
        value = value.bmm(self.value_projection).view(-1, seq_len, self.dim_v)
        key = key.bmm(self.key_projection).view(-1, seq_len, self.dim_q_k)
        query = query.bmm(self.query_projection).view(-1, q_len, self.dim_q_k)
        # (n_heads * batch, q_len, dim_v)
        context, _ = self.attention(value, key, query, mask)

        # # (n_heads * batch, q_len, dim_v) -> (batch * q_len, n_heads, dim_v) -> (batch, q_len, n_heads * dim_v)
        # context = context.view(self.n_heads, -1, self.dim_v).transpose(0, 1).view(-1, q_len, self.n_heads * self.dim_v)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)
        '''
        print("out:", out.shape)
        print(out)
        print("residual:", residual.shape)
        print(residual)
        '''
        new_out = out + residual
        return self.layer_normalization(new_out)

    def lexical_shortcut(self, embeddings, key, value):
        #(batch, seq_len, emb_size) -> (batch, seq_len, dim_m)
        K_sc = self.ek_normal(embeddings)
        V_sc = self.ev_normal(embeddings)
        # (batch, seq_len, dim_m) -> (batch, seq_len, dim_m)
        K_l = self.hk_normal(key)
        V_l = self.hv_normal(value)
        ##(batch, seq_len, dim_m) -> (batch, seq_len, dim_m)
        r_k = torch.sigmoid(self.gate_k(K_sc + K_l))
        r_v = torch.sigmoid(self.gate_v(V_sc + V_l))

        K_new = r_k.mul(K_sc) + (1 - r_k).mul(K_l)
        V_new = r_v.mul(V_sc) + (1 - r_v).mul(V_l)

        return K_new, V_new

    def stack_mask(self, mask):
        """Prepare mask tensor for multi-head Scaled Dot-Product Attention.

        Args:
            mask: A byte tensor of shape `(batch, q_len, seq_len)`.

        Returns:
            A byte tensor of shape `(n_heads * batch, q_len, seq_len)`.
        """
        return mask.repeat(self.n_heads, 1, 1)

    def stack_heads(self, tensor):
        """Prepare tensor for multi-head projection.

        Args:
            tensor: A float input tensor of shape `(batch, x, dim_m)`.

        Returns:
            Stacked input tensor n_head times of shape `(n_heads, batch * x, dim_m)`.
        """
        return tensor.contiguous().view(-1, self.dim_m).repeat(self.n_heads, 1, 1)


class luong_gate_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context

    def forward(self, h, selfatt=True):
        if selfatt:
            gamma_enc = self.linear_enc(self.context)  # Batch_size * Length * Hidden_size
            gamma_h = gamma_enc.transpose(1, 2)  # Batch_size * Hidden_size * Length
            weights = torch.bmm(gamma_enc, gamma_h)  # Batch_size * Length * Length
            weights = self.softmax(weights / math.sqrt(512))
            c_t = torch.bmm(weights, gamma_enc)  # Batch_size * Length * Hidden_size
            output = self.linear_out(torch.cat([gamma_enc, c_t], 2)) + self.context
            output = output.transpose(0, 1)  # Length * Batch_size * Hidden_size
        else:
            gamma_h = self.linear_in(h).unsqueeze(2)
            weights = torch.bmm(self.context, gamma_h).squeeze(2)
            weights = self.softmax(weights)
            c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights
