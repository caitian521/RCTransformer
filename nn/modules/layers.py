import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from nn.modules import MultiHeadAttention, PositionWise, luong_gate_attention, ScaledDotProductAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch=False):
        """Transformer encoder layer.

        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.

        Inputs:
            - **input** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `enc_seq_len` is length of encoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.

        Outputs:
            - **output** of shape `(batch, seq_len, dim_m)`, a float tensor.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=lexical_switch)
        self.positionwise = PositionWise(dim_m, dim_i, dropout)

    def forward(self, input, embeddings=None):
        enc_att = self.attention(input, input, input, embeddings=embeddings)
        output = self.positionwise(enc_att)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch):
        """Transformer decoder layer.

        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.

        Inputs:
            - **input** of shape `(batch, dec_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `dec_seq_len` is length of decoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.
            - **encoder_output** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `enc_seq_len` is length
              of encoder sequence.
            - **mask** of shape `(batch, dec_seq_len, dec_sec_len)`, a byte tensor containing mask for
              illegal connections between encoder and decoder sequence tokens. It's used to preserving
              the auto-regressive property.

        Outputs:
            - **output** of shape `(batch, dec_seq_len, dim_m)`, a float tensor.
        """
        super(TransformerDecoderLayer, self).__init__()

        self.masked_attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=lexical_switch)
        self.attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=False)
        self.positionwise = PositionWise(dim_m, dim_i, dropout)

    def forward(self, input, encoder_output, mask, embeddings=None):
        #print("decode start")
        dec_att = self.masked_attention(input, input, input, mask, embeddings)
        adj_att = self.attention(value=encoder_output, key=encoder_output, query=dec_att)
        output = self.positionwise(adj_att)

        return output


class CQAttention(nn.Module):
    def __init__(self, dim_m, dropout=0.1):
        super(CQAttention, self).__init__()
        w = torch.empty(dim_m * 3)
        lim = 1 / dim_m
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        self.dropout = dropout
        self.out_linear = nn.Linear(4*dim_m, dim_m)

    def mask_logits(self, target, mask):
        return target * mask + (1 - mask) * (-1e30)

    def forward(self, C, Q, cmask, qmask):

        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        S1 = F.softmax(self.mask_logits(S, qmask), dim=2)
        S2 = F.softmax(self.mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        #out = torch.cat([C, A, torch.mul(C, A)], dim=2)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        output = self.out_linear(out)
        return output


class TransformEncoder(nn.Module):
    def __init__(self, batch_size=16, emb_size=250, n_layers=6, dim_m=512, dim_q_k=64, dim_v=64, n_heads=8,
                 dim_i=2048, dropout=0.1, lexical_switch=False, topic=False):
        super(TransformEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout,
                                     lexical_switch) for i in range(n_layers)])
        self.topic = topic
        if topic:
            self.middle_layer = CQAttention(dim_m)

        self.encoder_state = None

    def forward(self, encoder_state, source_embeddings=None, topic_state=None, enc_mask=None, topic_mask=None):
        if encoder_state is not None:
            self.encoder_state = encoder_state
            for enc_layer in self.encoder_layers:
                self.encoder_state = enc_layer(self.encoder_state, source_embeddings)#, source_embeddings)


        if self.topic:
            for enc_layer in self.encoder_layers:
                topic_state = enc_layer(topic_state)

            middle_state = self.middle_layer(self.encoder_state, topic_state, enc_mask, topic_mask)
            return middle_state
        return self.encoder_state

    def reset_encoder_state(self):
        self.encoder_state = None


class TransformDecoder(nn.Module):
    def __init__(self, batch_size=16, emb_size=250, n_layers=6, dim_m=512, dim_q_k=64, dim_v=64, n_heads=8,
                 dim_i=2048, dropout=0.1, lexical_switch=False):
        super(TransformDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout,
                                     lexical_switch) for i in range(n_layers)])
        # It's original approach from paper, but i think it's better to use smooth transition from dim_m to vocab_size

    def forward(self, dec_state, middle_state, mask, target_embeddings=None):
        for dec_layer in self.decoder_layers:
            dec_state = dec_layer(dec_state, middle_state, mask, target_embeddings)

        return dec_state


class WordProbLayer(nn.Module):
    def __init__(self, dim_m, vocab_size, emb_share_tgt_prj=None,emb_align=None):
        super(WordProbLayer, self).__init__()
        #here can do copy or coverage mechanism
        if emb_share_tgt_prj is None:
            self.out = nn.Linear(dim_m, vocab_size, bias=False)
        else: #share embedding with input
            emb_size = emb_share_tgt_prj.shape[1]
            self.output_resize = nn.Linear(dim_m, emb_size, bias=False)
            self.tgt_word_proj = nn.Linear(emb_size, vocab_size, bias=False)
            self.tgt_word_proj.weight = emb_share_tgt_prj
            self.output_resize.weight = emb_align
            self.out = nn.Sequential(self.output_resize, self.tgt_word_proj, nn.Tanh())
            #self.logit_scale = (d_model ** -0.5)
            # self.logit_scale = 1.
        #self.softmax = F.softmax(dim=-1)

    def forward(self, dec_state):
        output = self.out(dec_state)
        return output

'''
===config need===
src_vocab_size
emb_size
hidden_size
enc_num_layers
dropout
'''

class GlobalEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, global_layers, cell='gru', embeddings=None, dropout=0.1,
                 bidirectional=True, encoding_gate=True, inception=True, gtu=False):
        super(GlobalEncoder, self).__init__()

        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            emb_size = embeddings.shape[1]
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.embedding.weight = nn.Parameter(embeddings, requires_grad=False)

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.inception = inception
        if self.inception:
            self.sw1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0),
                                     nn.BatchNorm1d(hidden_size), nn.ReLU())
            self.sw3 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0),
                                     nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                                     nn.ReLU(), nn.BatchNorm1d(hidden_size))
            self.sw33 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0),
                                      nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                      nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                                      nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                      nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                                      nn.ReLU(), nn.BatchNorm1d(hidden_size))
            self.linear = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.GLU(),
                                        nn.Dropout(dropout))
            self.filter_linear = nn.Linear(3 * hidden_size, hidden_size)

        if cell == 'gru':
            self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
                              num_layers=global_layers, dropout=dropout,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                               num_layers=global_layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.encoding_gate = encoding_gate
        self.gtu = gtu
        if encoding_gate:
            self.sigmoid = nn.Sigmoid()
            if self.gtu:
                self.dropout = nn.Dropout(dropout)
                self.layer_normalization = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, inputs, lengths):
        inputs = inputs.t()
        lengths = lengths.tolist()
        embs = pack(self.embedding(inputs), lengths)
        self.rnn.flatten_parameters()
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Batch_size * Length * Hidden_size

        if self.inception:
            outputs = outputs.transpose(0, 1).transpose(1, 2)

            conv1 = self.sw1(outputs)
            conv3 = self.sw3(outputs)
            conv33 = self.sw33(outputs)
            conv = torch.cat((conv1, conv3, conv33), 1)

            conv = self.filter_linear(conv.transpose(1, 2))
            #conv = self.sw3(outputs).transpose(1, 2)
            outputs = outputs.transpose(1, 2).transpose(0, 1)   #seq_len, batch, dim
            outputs = outputs.transpose(0, 1)
            if self.encoding_gate:
                if self.gtu:
                    # conv =   "weight norm"
                    # outputs "weight norm"
                    gate = self.sigmoid(conv)
                    tan_conv = torch.tanh(outputs)
                    gtu_out = tan_conv * gate
                    return self.layer_normalization(gtu_out + outputs)
                else:
                    gate = self.sigmoid(conv)
                    outputs = outputs * gate
                    return outputs
            else:
                return conv
        else:
            return outputs.transpose(0, 1)

class LastDecoder(nn.Module):
    def __init__(self, dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch, stack=True):
        """Transformer decoder layer.

        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.

        Inputs:
            - **input** of shape `(batch, dec_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `dec_seq_len` is length of decoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.
            - **encoder_output** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `enc_seq_len` is length
              of encoder sequence.
            - **mask** of shape `(batch, dec_seq_len, dec_sec_len)`, a byte tensor containing mask for
              illegal connections between encoder and decoder sequence tokens. It's used to preserving
              the auto-regressive property.

        Outputs:
            - **output** of shape `(batch, dec_seq_len, dim_m)`, a float tensor.
        """
        super(LastDecoder, self).__init__()
        self.masked_attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=lexical_switch)
        self.attention = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=False)
        self.attention2 = MultiHeadAttention(n_heads, dim_m, dim_q_k, dim_v, batch_size, emb_size, dropout, lexical=False)
        self.positionwise = PositionWise(dim_m, dim_i, dropout)
        self.stack = stack
        if self.stack is False: #use gated sum mode
            self.two_encoder_mix = nn.Linear(2*dim_m, dim_m)

    def forward(self, input, encoder_output, global_output, mask, target_embeddings=None):
        #print("decode start")
        if self.stack:
            dec_att = self.masked_attention(input, input, input, mask, target_embeddings)
            adj_att = self.attention(value=encoder_output, key=encoder_output, query=dec_att)
            egd_att = self.attention2(value=global_output, key=global_output, query=adj_att)
            output = self.positionwise(egd_att)
        else:
            dec_att = self.masked_attention(input, input, input, mask, target_embeddings)
            adj_att = self.attention(value=encoder_output, key=encoder_output, query=dec_att)
            egd_att = self.attention2(value=global_output, key=global_output, query=dec_att)

            adj_egd_cat = torch.cat([adj_att, egd_att], dim=-1)
            two_encoder = self.two_encoder_mix(adj_egd_cat)
            gate = torch.sigmoid(two_encoder)
            output = gate.mul(adj_att) + (1 - gate).mul(egd_att)
            output = self.positionwise(output)
        return output
