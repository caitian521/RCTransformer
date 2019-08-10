import torch
import torch.nn.functional as F
from nn import TransformEncoder, TransformDecoder, LabelSmoothing, PositionalEmbedding, WordProbLayer, GlobalEncoder, LastDecoder
from .beam_search import *
from config import config
'''
vocab_size = 15000
emb_size = 250
model_dim = 512
enc_n_layers = 6
dec_n_layers = 6
n_heads = 8
inner_dim = 512
'''

class TransformerSummarizer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, initial_idx=2, embedding_weights=None, enc_n_layers=6, dec_n_layers=6, global_layers=2,
                 batch_size=16, emb_size=250, dim_m=512, n_heads=8, dim_i=2048, dropout=0.1, lexical_switch = True,
                 emb_share_prj=False, global_encoding=False, encoding_gate=False, stack=False, topic=False, inception=True, gtu=False):
        """Pure transformer model for summarization task. Actually, it's possible to use this model for MT task.

        Args:
            max_seq_len (int): maximum length of input sequences.
            vocab_size (int): vocabulary size.
            initial_idx (int, optional): initial token index.
            embedding_weights (torch.Tensor, optional): float tensor of shape `(vocab_size, dim_m)`, containing
                embedding weights. Embedding size value would inherited from shape of `embedding_weights` tensor.
            n_layers (int, optional): number transformer layers.
            emb_size (int, optional): embedding size. You do not need to specify a value if you are using
              embedding weights.
            dim_m (int, optional): model dimension (hidden or input size).
            n_heads (int, optional): number of attention heads.
            dim_i (int, optional): inner dimension of position-wise sublayer.
            dropout (float, optional): dropout probability.

        Input:
            - **source_seq** of shape `(batch, source_seq_len)`: a long tensor, containing token indexes of
              source sequence.
            - **target_seq** of shape `(batch, target_seq_len)`: (optional) a long tensor, containing token indexes of
              target sequence.
            - **max_target_seq_len** an int (optional): maximum length of generated sequence. If `target_seq` is None
              `max_target_seq_len` must be defined.

        Output:
            - **generated_seq_probs** of shape `(batch, target_seq_len, vocab_size)`: a float tensor, containing token
              probabilities.
            - **generated_seq** of shape `(batch, target_seq_len)`: a long tensor, containing generated token,
              determined by naive argmax encoding.

        Notes:
            - Model dimension `dim_m` must be divisible by `n_heads` without a remainder. It's necessary for calculating
              projection sizes for multi-head attention.
        """
        super(TransformerSummarizer, self).__init__()

        self.vocab_size = vocab_size
        self.initial_token_idx = initial_idx
        self.lexical_switch = lexical_switch
        self.global_encoding = global_encoding
        self.topic = topic


        assert dim_m % n_heads == 0, 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        dim_proj = dim_m // n_heads
        self.positional_encoding = PositionalEmbedding(max_seq_len, dim_m, vocab_size, emb_size, embedding_weights)

        self.encoder = TransformEncoder(batch_size, emb_size, enc_n_layers, dim_m,
                                       dim_proj, dim_proj, n_heads, dim_i, dropout, lexical_switch, topic)

        self.decoder = TransformDecoder(batch_size, emb_size, dec_n_layers, dim_m,
                                       dim_proj, dim_proj, n_heads, dim_i, dropout, lexical_switch)

        if global_encoding:
            self.global_encoder = GlobalEncoder(vocab_size, emb_size, dim_m, global_layers, embeddings=embedding_weights,
                                                encoding_gate=encoding_gate, inception=inception, gtu=gtu)
            self.decoder_last = LastDecoder(dim_m, dim_proj, dim_proj, n_heads, dim_i, batch_size, emb_size, dropout, lexical_switch, stack=stack)

        if emb_share_prj:
            emb_share_tgt_prj = self.positional_encoding.embedding.weight
            emb_align = self.positional_encoding.alignment.weight
        else:
            emb_share_tgt_prj = None
            emb_align = None

        self.word_prob = WordProbLayer(dim_m, vocab_size, emb_share_tgt_prj, emb_align)
        # Get initial probabilities for bos token.
        self.initial_probs = self.get_initial_probs(vocab_size, initial_idx)

        self.label_smooth = LabelSmoothing(vocab_size)


    def forward(self, source_seq, target_seq, source_lens=None, topic_seq=None, eval_flag = False):
        self.encoder.reset_encoder_state()
        encoder_state = self.positional_encoding(source_seq)
        dec_state = self.positional_encoding(target_seq)

        if self.lexical_switch is False:
            source_embeddings, target_embeddings = None, None
        else:
            source_embeddings, target_embeddings = encoder_state, dec_state

        enc_mask = (torch.zeros_like(source_seq) != source_seq).float()

        if self.topic:
            topic_state = self.positional_encoding(topic_seq, topic_flag = True)
            topic_mask = (torch.zeros_like(topic_seq) != topic_seq).float()
            middle_state = self.encoder(encoder_state, source_embeddings, topic_state, enc_mask, topic_mask)
        else:
            middle_state = self.encoder(encoder_state, source_embeddings)

        mask = self.autoregressive_mask(target_seq)
        dec_out = self.decoder(dec_state, middle_state, mask, target_embeddings=target_embeddings)

        if self.global_encoding:
            global_state = self.global_encoder(source_seq, source_lens)
            dec_out = self.decoder_last(dec_out, middle_state, global_state, mask, target_embeddings=target_embeddings)

        output = self.word_prob(dec_out)

        if eval_flag:
            return output
        batch_size = source_seq.shape[0]
        probs = torch.cat((self.initial_probs.to(source_seq.device).repeat(batch_size, 1, 1), output[:, :-1, :]), dim=1)
        seq = probs.argmax(-1)

        loss = self.label_smooth(probs.view(-1, self.vocab_size), target_seq.view(-1)).cuda()
        if config.multi_gpu:
            loss = loss.mean()
        return loss, seq


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


    def inference_greedy(self, source_seq, source_length=None, target_seq=None, topic_seq=None, max_target_seq_len=None):
        batch_size = source_seq.shape[0]

        if target_seq is not None:
            target_seq_len = target_seq.shape[1]
        else:
            assert max_target_seq_len is not None, 'Target sequence length don\'t defined'
            target_seq_len = max_target_seq_len

        # Create initial tokens.
        generated_seq = torch.full((batch_size, 1), self.initial_token_idx, dtype=torch.long, device=source_seq.device)

        # It's very important to do this before every train batch cycle.
        self.encoder.reset_encoder_state()
        for i in range(1, target_seq_len):
            #source_seq, target_seq, source_lens=None, topic_seq=None, eval_flag = False
            output = self.forward(source_seq, generated_seq, source_length, topic_seq, eval_flag=True)

            if target_seq is None:
                # Take last token probabilities and find it's index.
                generated_token_idx = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            else:
                # Use target sequence for next initial words.
                generated_token_idx = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                # generated_token_idx = target_seq[:, i].unsqueeze(1)

            # Concatenate generated token with sequence.
            generated_seq = torch.cat((generated_seq, generated_token_idx), dim=-1)

        generated_seq_probs = torch.cat((self.initial_probs.to(source_seq.device).repeat(batch_size, 1, 1), output),
                                        dim=1)
        return generated_seq_probs, generated_seq

    def train_step(self, batch, optim):
        """Make train step.

        Args:
            batch (data.Batch): batch.
            optim (torch.optim.Optimizer): optimizer.

        Returns:
            float: loss value.

        """
        self.train()
        optim.zero_grad()
        #loss, seq = self.forward(batch.src, batch.trg)


        output = self.forward(batch.src, batch.trg, eval_flag=True)
        #print(output.shape)
        batch_size = output.shape[0]
        probs = torch.cat((self.initial_probs.to(batch.src.device).repeat(batch_size, 1, 1), output[:, :-1, :]), dim=1)
        seq = probs.argmax(-1)

        loss = self.label_smooth(probs.view(-1, self.vocab_size), batch.trg.view(-1)).cuda()
        if config.multi_gpu:
            loss = loss.mean()

        loss.backward()
        optim.step()

        return loss.item(), seq

    def inference_beam(self, source_seq, source_length=None, max_len=15, beam_size=5, topic_seq=None):
        # Calculate encoder state for batch.
        self.encoder.reset_encoder_state()
        enc_mask = (torch.zeros_like(source_seq) != source_seq).float()
        #print(source_seq)
        encoder_state = self.positional_encoding(source_seq)
        if self.lexical_switch:
            source_embeddings = encoder_state
        else:
            source_embeddings=None

        if self.topic:
            topic_mask = (torch.zeros_like(topic_seq) != topic_seq).float()
            topic_state = self.positional_encoding(topic_seq, topic_flag=True)
            encoder_state = self.encoder(encoder_state, source_embeddings, topic_state, enc_mask, topic_mask)
        else:
            encoder_state = self.encoder(encoder_state,source_embeddings)
        if self.global_encoding:
            global_state = self.global_encoder(source_seq, source_length)
        else:
            global_state = None

        seq = beam_search(self, encoder_state, global_state,  max_len, encoder_state.shape[-1], beam_size, start_id=2, end_id=3)
        return seq

    def evaluate(self, src, lengths, max_seq_len=None, beam_size=None, topic_seq=None, trg=None):
        """Evaluate model.

        Args:
            batch (data.Batch): Evaluated batch.

        Returns:
            float: loss value.

        """
        self.eval()
        with torch.no_grad():

            if beam_size is None:
                probs, seq = self.inference_greedy(src, lengths, target_seq=trg, max_target_seq_len=max_seq_len, topic_seq=topic_seq)
                if trg is not None:
                    loss = self.label_smooth(probs.contiguous().view(-1, self.vocab_size), trg.view(-1))
                    loss = loss.item()
                else:
                    loss = None
            else:
                seq = self.inference_beam(src, lengths, max_len=max_seq_len, beam_size=beam_size, topic_seq=topic_seq)
                loss = None
        return seq, loss

    def sample(self, batch, max_seq_len=None):
        """Generate sample.

        Args:
            batch (data.Batch): Sample batch.
            max_seq_len (int, optional): Maximum length of generated summary.

        Returns:
            torch.Tensor: long tensor of shape `(batch, target_seq_len)`, containing generated sequences.

        """
        self.eval()

        if max_seq_len is None:
            max_seq_len = batch.trg.shape[1]

        with torch.no_grad():
            probs, seq = self.inference_beam(batch.src, max_seq_len, beam_size=5, topic_seq=None)
        # It's better to use beam search, I guess.
        # TODO
        return seq

    def learnable_parameters(self):
        """Get all learnable parameters of the model.

        Returns: Generator of parameters.

        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    @staticmethod
    def get_initial_probs(vocab_size, initial_token_idx):
        """Generate initial probability distribution for vocabulary.

        Args:
            vocab_size (int): Size of vocabulary.
            initial_token_idx (int): Initial token index.

        Returns:
            torch.Tensor: float tensor of shape `(1, vocab_size)`.

        """
        probs = torch.zeros(1, vocab_size)
        probs[0, initial_token_idx] = 1
        return probs.float()


