import argparse
import logging
import os
import datetime
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from rouge import Rouge
import json
from torch.distributions import Categorical
import torch.nn.functional as F
from data import DataLoader, CNNDailyMail
from model import TransformerSummarizer
from config import config
os.environ["CUDA_VISIBLE_DEVICES"] = " 0,1,2,3"
timestamp = datetime.datetime.now().strftime("%m-%d-%H")
data_type = "train"

class Train(object):
    def __init__(self, args):
        logging.info('Loading dataset')
        #self.loader = CNNDailyMail(config.dataset, data_type, ['src', 'trg'], config.bpe_model_filename)
        self.loader = DataLoader(config.dataset, data_type, ['src', 'trg'], config.bpe_model_filename)
        logging.info('Dataset has been loaded.Total size: %s, maxlen: %s', self.loader.data_length, self.loader.max_len)

        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        if config.multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        self.writer = SummaryWriter(config.log + config.prefix)

        if args.pretrain_emb:
            self.embeddings = torch.from_numpy(np.load(config.emb_filename)).float()
            logging.info('Use vocabulary and embedding sizes from embedding dump.')
            self.vocab_size, self.emb_size = self.embeddings.shape
        else:
            self.embeddings = None
            self.vocab_size, self.emb_size = config.vocab_size, config.emb_size

        self.args = args
        self.m_args = {'max_seq_len': self.loader.max_len, 'vocab_size': self.vocab_size,
                       'n_layers': config.n_layers, 'batch_size': config.train_bs, 'emb_size': self.emb_size,
                       'dim_m': config.model_dim,
                       'n_heads': config.n_heads, 'dim_i': config.inner_dim, 'dropout': config.dropout,
                       'embedding_weights': self.embeddings, 'lexical_switch': args.lexical}
        #if config.multi_gpu:
        #    self.m_args['batch_size'] *= n_gpu
        self.error_file = open(config.error_filename, "w")

    def setup_train(self):
        logging.info('Create model')

        self.model = TransformerSummarizer(**self.m_args).to(self.device)
        if config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2,3])
        # self.model = get_cuda(self.model)
        self.m_args['embedding_weights'] = None
        optim_param = self.model.module.learnable_parameters() if isinstance(self.model, torch.nn.DataParallel) else self.model.learnable_parameters()
        self.optimizer = Adam(optim_param, lr=config.learning_rate, amsgrad=True,
                              betas=[0.9, 0.98], eps=1e-9)
        start_iter = 0

        if self.args.load_model is not None:
            load_model_path = os.path.join(config.save_model_path, self.args.load_model)
            checkpoint = torch.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.module.load_state_dict(checkpoint["model_dict"])
            self.optimizer.load_state_dict(checkpoint["trainer_dict"])
            logging.info("Loaded model at " + load_model_path)

        if self.args.new_lr is not None:
            self.optimizer = Adam(self.model.module.learnable_parameters(), lr=self.args.new_lr)
        '''
        model_dict = self.model.module.state_dict()
        for k, v in model_dict.items():
            print(k)
        '''

        return start_iter

    def train_RL(self, batch, optim, mle_weight):

        batch_size = batch.src.shape[0]
        source_seq, target_seq = batch.src, batch.trg
        extra_zeros, enc_batch_extend_vocab, trg_extend = batch.extra_zeros, batch.enc_batch_extend_vocab, batch.trg_extend_vocab

        if self.args.topic:
            greedy_seq, out_oov, probs, out_probs, mle_loss = self.model.forward(source_seq, target_seq, extra_zeros,
                                                                                 enc_batch_extend_vocab, trg_extend, topic_seq=batch.topic)
        else:
            greedy_seq, out_oov, probs, out_probs, mle_loss = self.model.forward(source_seq, target_seq, extra_zeros, enc_batch_extend_vocab, trg_extend)

        # multinomial sampling
        probs = F.softmax(probs, dim=-1)
        multi_dist = Categorical(probs)
        sample_seq = multi_dist.sample()  # perform multinomial sampling
        index = sample_seq.view(batch_size, -1, 1)
        sample_prob = torch.gather(probs, 2, index) #batch, seq_len, 1

        non_zero = (sample_prob == self.loader.eos_idx)
        mask = np.zeros_like(non_zero.cpu())
        for i in range(non_zero.shape[0]):
            index = torch.nonzero(non_zero[i])
            #print(index)
            if index.shape[0] == 0:
                mask[i] = 1
            else:
                mask[i][:index[0]] = 1

        mask = torch.FloatTensor(mask).cuda()
        lens = torch.sum(mask, dim=1) + 1# Length of sampled sentence
        RL_logs = torch.sum(mask * sample_prob, dim=1) / lens.cuda()

        #compute normalizied log probability of a sentence
        #print(sample_seq)
        sample_seq = self.loader.decode_oov(greedy_seq, source_oovs=batch.source_oov, oov=sample_seq)
        generated = self.loader.decode_oov(greedy_seq, source_oovs=batch.source_oov, oov=out_oov)
        original = batch.trg_text

        sample_reward = self.reward_function(sample_seq, original)
        baseline_reward = self.reward_function(generated, original)
        sample_reward = torch.FloatTensor(sample_reward).cuda()
        baseline_reward = torch.FloatTensor(baseline_reward).cuda()
        #print(RL_logs)
        #print("reward", sample_reward, baseline_reward)
        # if iter%200 == 0:
        #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
        # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
        rl_loss = -(sample_reward - baseline_reward) * RL_logs
        rl_loss = torch.mean(rl_loss)

        batch_reward = torch.mean(sample_reward).item()
        #rl_loss = T.FloatTensor([0]).cuda()
        #batch_reward = 0

        # ------------------------------------------------------------------------------------

        if config.multi_gpu:
            mle_loss = mle_loss.mean()
            rl_loss = rl_loss.mean()
        #print("mle loss, rl loss, reward:", mle_loss.item(), rl_loss.item(), batch_reward)
        optim.zero_grad()
        (mle_weight * mle_loss + (1-mle_weight) * rl_loss).backward()
        optim.step()

        #mix_loss = mle_weight * mle_loss + (1-mle_weight) * rl_loss
        return mle_loss.item(), generated, original, batch_reward


    def reward_function(self, decoded_sents, original_sents):
        #print(decoded_sents)
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            print("Rouge failed for multi sentence evaluation.. Finding exact pair")
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                except Exception:
                    #print("Error occured at:")
                    #print("decoded_sents:", decoded_sents[i])
                    #print("original_sents:", original_sents[i])
                    score = [{"rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        return rouge_l_f1


    def save_model(self, iter):
        save_path = config.dump_filename + "/%07d.tar" % iter
        torch.save({
            "iter": iter + 1,
            "model_dict": self.model.module.state_dict() if config.multi_gpu else self.model.state_dict(),
            "trainer_dict": self.optimizer.state_dict()
        }, save_path)
        with open(config.args_filename, 'w') as f:
            f.write(json.dumps(self.m_args))
        logging.info('Model has been saved')

    def trainIters(self):
        logging.info('Start training')
        start_iter = self.setup_train()
        count = loss_total = rouge_total = 0
        record_rouge = 0.0
        self.model.train()

        for i in range(start_iter, config.iters):
            train_batch = self.loader.next_batch(config.train_bs, self.device)
            self.optimizer.zero_grad()
            source_seq, target_seq = train_batch.src, train_batch.trg
            extra_zeros, enc_batch_extend_vocab, trg_extend = train_batch.extra_zeros, train_batch.enc_batch_extend_vocab, train_batch.trg_extend_vocab
            if self.args.topic:
                seq, loss, out_oov, out_copy_probs, out_probs = self.model.forward(source_seq, target_seq, extra_zeros, enc_batch_extend_vocab, trg_extend, topic_seq=train_batch.topic)
            else:
                seq, loss, out_oov, out_copy_probs, out_probs = self.model.forward(source_seq, target_seq, extra_zeros, enc_batch_extend_vocab, trg_extend)

            if self.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            #print(loss)
            #generated = self.loader.decode(seq)
            generated = self.loader.decode_oov(seq, source_oovs=train_batch.source_oov, oov=out_oov)
            original = train_batch.trg_text
            '''
            print(generated[0])
            print(original[0])
            print(train_batch.src_text[0])
            '''
            scores = self.reward_function(generated, original)
            scores = np.array(scores).mean()

            loss_total += loss
            rouge_total += scores
            count += 1
            if i % config.train_interval == 0:
                loss_avg = loss_total / count
                rouge_avg = rouge_total / count
                logging.info('Iteration %d; Loss: %f; rouge: %.4f; loss avg: %.4f; rouge avg: %.4f', i, loss, scores,
                             loss_avg, rouge_avg)
                self.writer.add_scalar('Loss', loss, i)
                loss_total = rouge_total = count = 0
            if i % config.save_interval == 0 and i>=25000:
                if scores > record_rouge:
                    self.save_model(i)
                    record_rouge = scores

        for i in range(config.iters, config.rl_iters):
            train_batch = self.loader.next_batch(config.train_bs, self.device)
            loss, generated, original, reward = self.train_RL(train_batch, self.optimizer, config.mle_weight)
            #generated = self.loader.decode(seq)
            #original = self.loader.decode(train_batch.trg)

            scores = self.reward_function(generated, original)
            scores = np.array(scores).mean()

            loss_total += loss
            rouge_total += reward
            count += 1
            if i % config.train_interval == 0:
                loss_avg = loss_total / count
                rouge_avg = rouge_total / count
                logging.info('Iteration %d; Loss: %f; rouge: %.4f; loss avg: %.4f; reward avg: %.4f', i, loss, scores,
                             loss_avg, rouge_avg)
                self.writer.add_scalar('Loss', loss, i)
                loss_total = rouge_total = count = 0
            if i % config.save_interval == 0 and i >= 25000:
                    self.save_model(i)


if __name__ == "__main__":
    description = 'NN model for abstractive summarization based on transformer model.'
    epilog = 'Model training can be performed in two modes: with pretrained embeddings or train embeddings with model ' \
             'simultaneously. To choose mode use --pretrain_emb argument. ' \
             'Please, use deeper model configuration than this, if you want obtain good results.'

    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--lexical', action='store_true', help='whether to use lexical shortcut')
    parser.add_argument('--topic', action='store_true', help='whether to use topic attention')
    parser.add_argument('--fine_tune', action='store_true', help='whether to use fine tune transformer')
    parser.add_argument('--pretrain_emb', action='store_true', help='use pretrained embeddings')
    parser.add_argument("--load_model", type=str, default=None) #"0073000.tar")
    parser.add_argument('--new_lr', type=float, default=None)

    args = parser.parse_args()

    print(config.this_model_path)
    if not os.path.exists(config.this_model_path):
        os.makedirs(config.this_model_path)
    print("")
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.log_filename),
            logging.StreamHandler()
        ])

    if torch.cuda.is_available():
        if not args.cuda:
            logging.info('You have a CUDA device, so you should probably run with --cuda')
    else:
        if args.cuda:
            logging.warninig('You have no CUDA device. Start learning on CPU.')

    train_processor = Train(args)
    train_processor.trainIters()
