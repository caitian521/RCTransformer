import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Beam(nn.Module):
    def __init__(self, middle_state, global_state, beam_size, start_id, end_id):
        self.beam_size = beam_size
        #for giga sorpus start_id = 2  end_id  = 3 cnn 102,103
        self.start_id = start_id  #2
        self.end_id = end_id
        self.pad_id = 0
        self.tokens = T.LongTensor(self.beam_size, 1).fill_(self.pad_id)  # (beam, t) after t time steps
        self.tokens[0,0] = self.start_id
        self.scores = T.FloatTensor(self.beam_size, 1).fill_(-30)  # beam,1; Initial score of beams = -30
        self.tokens = self.get_cuda(self.tokens)
        self.scores = self.get_cuda(self.scores)
        self.middle_state = middle_state.unsqueeze(0).repeat(self.beam_size, 1, 1)  # beam,seq_len, dim
        if global_state is None:
            self.global_state = None
        else:
            self.global_state = global_state.unsqueeze(0).repeat(self.beam_size, 1, 1)  # beam,seq_len, dim

        self.done = False

    def update(self, output):
        '''
        output: beam, t, vocab_size

        '''
        log_probs = output[:,-1,:].squeeze(1)  #beam, vocab_size

        vocab_size = output.shape[-1]

        scores = log_probs + self.scores # beam, vocab_size
        scores = scores.view(-1, 1)  # beam*vocab_size, 1
        best_scores, best_scores_id = T.topk(input=scores, k=self.beam_size,
                                             dim=0)  # will be sorted in descending order of scores
        self.scores = best_scores  # (beam,1); sorted

        beams_order = best_scores_id.squeeze(1) / vocab_size  # (beam,); sorted
        best_words = best_scores_id % vocab_size  # (beam,1); sorted

        self.tokens = self.tokens[beams_order]  # (beam, t); sorted
        self.tokens = T.cat([self.tokens, best_words], dim=1)  # (beam, t+1); sorted

        # End condition is when top-of-beam is EOS.
        if best_words[0][0] == self.end_id:
            self.done = True

    def get_best(self):
        best_token = self.tokens[0].cpu().numpy().tolist()              #Since beams are always in sorted (descending) order, 1st beam is the best beam
        try:
            end_idx = best_token.index(self.end_id)
        except ValueError:
            end_idx = len(best_token)
        best_token = best_token[1:end_idx]
        return best_token

    def get_cuda(self, tensor):
        if T.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def __str__(self):
        return 'p = {}, data = {}'.format(self.prob, self.data)


def beam_search(model, middle_state, global_state, max_len, dim, beam_size, start_id, end_id):
    enc_len = middle_state.shape[1]
    batch_size = middle_state.shape[0]
    beam_idx = list(range(batch_size))
    if global_state is None:
        beams = [Beam(middle_state[i], None, beam_size, start_id, end_id) for i in range(batch_size)]   #For each example in batch, create Beam object
    else:
        beams = [Beam(middle_state[i], global_state[i], beam_size, start_id, end_id) for i in range(batch_size)]   #For each example in batch, create Beam object
    n_rem = batch_size
    for t in range(max_len):
        generated_seq = T.stack([beam.tokens for beam in beams if beam.done==False]).contiguous().view(-1, t+1) #(rem*beam, t)
        dec_state = model.positional_encoding(generated_seq)      #rem*beam, t, dim  rem:remaining batch

        mask = model.autoregressive_mask(generated_seq)   # (batch, seq_len, )  (batch, seq_len, seq_len)--> rem*beam, t, t

        middle_state = T.stack([beam.middle_state for beam in beams if beam.done==False]).contiguous().view(-1,enc_len,dim) #rem*beam,enc_len,dim

        if model.lexical_switch:
            target_embeddings = dec_state
        else:
            target_embeddings = None
        dec_out = model.decoder(dec_state, middle_state, mask, target_embeddings) #rem*beam, t, dim

        if global_state is not None:
            global_state = T.stack([beam.global_state for beam in beams if beam.done==False]).contiguous().view(-1,enc_len,dim) #rem*beam,enc_len,dim
            dec_out = model.decoder_last(dec_out, middle_state, global_state, mask, target_embeddings)

        output = model.word_prob(dec_out)  #rem*beam, t, vocab_size
        output = F.log_softmax(output, dim=-1)
        output = output.view(n_rem, beam_size, t+1, -1)  #rem, beam, t, vocab_size

        active = []  # indices of active beams after beam search

        for i in range(n_rem):
            b = beam_idx[i]
            beam = beams[b]
            if beam.done:
                continue
            beam.update(output[i])
            if beam.done == False:
                active.append(b)

        if len(active) == 0:
            break
        beam_idx = active
        n_rem = len(beam_idx)

    predicted_seqs = []
    for beam in beams:
        predicted_seqs.append(beam.get_best())

    return predicted_seqs















