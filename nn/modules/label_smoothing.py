import torch
import torch.nn as nn

import torch.nn.functional as F
from nn.modules import TransformerEncoderLayer, TransformerDecoderLayer
from nn.modules import PositionalEmbedding, CQAttention
'''
class Transformer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, batch_size=16, emb_size=250, embeddings=None, n_layers=6, dim_m=512, dim_q_k=64, dim_v=64, n_heads=8,
                 dim_i=2048, dropout=0.1, lexical_switch=False):
        """Transformer model from 'Attention Is All You Need' paper.

        Args:
            max_seq_len (int): Maximum sequence length.
            vocab_size (int): Vocabulary size.
            emb_size (int, optional): Embedding size. You do not need to specify a value if you are using
              embedding weights.
            embeddings (torch.Tensor, optional): Long tensor of shape `(vocab_size, emb_size)` - embedding tensor.
              Embedding size value would inherited from shape of this tensor.
            n_layers (int, optional): Number of transformer layers.
            dim_m (int, optional): Model hidden size, must be equal with embedding size.
            dim_q_k (int, optional): Dimension of `query` & `key` attention projections.
            dim_v (int, optional): Dimension of `value` attention projection.
            n_heads (int, optional): Number of attention heads.
            dim_i (int, optional): Inner dimension of feed-forward position-wise sublayer.
            dropout (float, optional): Dropout probability.

        Variables:
            - **encoder_state**: a float tensor of shape `(batch, enc_seq_len, dim_m)` containing encoder state from
              last layer.

        Inputs:
            - **enc_seq** of shape `(batch, enc_seq_len)`, a long tensor encoder input sequence.
            - **dec_seq** of shape `(batch, dec_seq_len)`, a long tensor decoder input sequence.

        Outputs:
            - **output** of of shape `(batch, dec_seq_len, vocab_size)`, a float tensor of vocabulary probability
              distribution.

        Notes:
            - For optimizing model, encoder state stores in local variable and calculate only one per batch. After
              auto-regressive process encoder state must be reset. You can do this using
              :func:`Transformer.reset_encoder_state`.
        """
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEmbedding(max_seq_len, dim_m, vocab_size, emb_size, embeddings)
        #self.topic_encoder = TopicEncoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch) for i in range(n_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch) for i in range(n_layers)])
        # It's original approach from paper, but i think it's better to use smooth transition from dim_m to vocab_size
        self.out = nn.Linear(dim_m, vocab_size)
        self.softmax = nn.Softmax(-1)

        self.middle_layer = CQAttention(dim_m)

        self.encoder_state = None

    def forward(self, enc_seq, dec_seq, topic_seq=None):
        # Calculate encoder state for batch.
        #topic_state = self.positional_encoding(topic_seq, topic_flag = True)

        if self.encoder_state is None:
            # Sum embeddings with positional encodings.
            self.encoder_state, source_embeddings = self.positional_encoding(enc_seq)
            for enc_layer in self.encoder_layers:
                self.encoder_state = enc_layer(self.encoder_state)#, source_embeddings)

        #for enc_layer in self.encoder_layers:
        #    topic_state = enc_layer(topic_state)

        #topic_mask = (torch.zeros_like(topic_seq) != topic_seq).float()
        #enc_mask = (torch.zeros_like(enc_seq) != enc_seq).float()
        #middle_state  = self.middle_layer(self.encoder_state, topic_state, enc_mask, topic_mask)
        #print(middle_state.shape)
        # Decoder block.
        # Apply positional encoding.
        dec_state, target_embeddings = self.positional_encoding(dec_seq)

        mask = self.autoregressive_mask(dec_seq)

        for dec_layer in self.decoder_layers:
            #dec_state = dec_layer(dec_state, middle_state, mask, target_embeddings)
            dec_state = dec_layer(dec_state, self.encoder_state, mask)

        output = self.out(dec_state)

        return output

    def reset_encoder_state(self):
        """Reset previous encoder state of batch. This method must calls before process new batch.
        """
        self.encoder_state = None

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto-regressive mask for tensor. It's used to preserving the auto-regressive property.

        Args:
            tensor (torch.Tensor): of shape `(batch, seq_len, dim)`.

        Returns:
            torch.Tensor: a byte mask tensor of shape `(batch, seq_len, seq_len)` containing mask for
            illegal attention connections between decoder sequence tokens.

        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()
'''
class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.1):
        super(LabelSmoothing, self).__init__()

        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.pad_index = 0

    def forward(self, x, target):
        '''
            x (batch*dec_seq_len, vocab_size),每一个类的概率log P
            target (batch* dec_seq_len,)
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
        '''

        one_hot = torch.zeros_like(x).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.size - 1)
        #log_prb = x
        log_prb = F.log_softmax(x, dim=1)
        #print(log_prb)
        non_pad_mask = target.ne(self.pad_index)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / non_pad_mask.sum()  #1.1354
        return loss

        '''
        x_log = nn.LogSoftmax(x, dim=1)

        mask = torch.nonzero(target.eq(0)).squeeze().long()
        dim = target.shape[0]  #batch_size*seq_len
        target = target.view(-1, 1)  # batch_size*seq_len,1

        y = torch.zeros(dim, self.size) #batch_size*seq_len, vocab_size
        y.fill_(self.smoothing / (self.size - 1))
        y.scatter_(1, target.cpu(), self.confidence)
        y.index_fill_(0,mask.cpu(),0)

        return -torch.sum(y.cuda() * x_log / dim)
        '''



if __name__ == "__main__":
    regular = LabelSmoothing(5)

    target = torch.LongTensor([[2, 1, 0],[3, 1, 0]])

    x = torch.FloatTensor([[[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.9, 0.2, 0.1, 0],
                            [1, 0.2, 0.7, 0.1, 0]],
                           [[0, 0.2, 0.1, 0.8, 0],
                            [0, 0.9, 0.1, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0.05]]])

    cost_value = regular(x.view(-1,5), target.view(-1))
    print(cost_value)