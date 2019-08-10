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
from model import TransformerSummarizer, Optim
from model import lr_scheduler as L

from config import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
timestamp = datetime.datetime.now().strftime("%m-%d-%H")
train_type = "train"
eval_type = "val"


class Train(object):
    def __init__(self, args):
        logging.info('Loading dataset')
        #self.loader = CNNDailyMail(config.dataset, data_type, ['src', 'trg'], config.bpe_model_filename)
        self.loader = DataLoader(config.dataset, train_type, ['src', 'trg'], config.bpe_model_filename)
        self.eval_loader = DataLoader(config.dataset, eval_type, ['src', 'trg'], config.bpe_model_filename)
        logging.info('Dataset has been loaded.Total size: %s, maxlen: %s', self.loader.data_length, self.loader.max_len)

        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        if config.multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        self.writer = SummaryWriter(config.dump_filename)

        if args.pretrain_emb:
            self.embeddings = torch.from_numpy(np.load(config.emb_filename)).float()
            logging.info('Use vocabulary and embedding sizes from embedding dump.')
            self.vocab_size, self.emb_size = self.embeddings.shape
        else:
            self.embeddings = None
            self.vocab_size, self.emb_size = config.vocab_size, config.emb_size

        self.args = args
        self.m_args = {'max_seq_len': self.loader.max_len, 'vocab_size': self.vocab_size,
                       'enc_n_layers': config.enc_n_layers, 'dec_n_layers': config.dec_n_layers, 'global_layers': config.global_layers,
                       'batch_size': config.train_bs, 'emb_size': self.emb_size, 'dim_m': config.model_dim,
                       'n_heads': config.n_heads, 'dim_i': config.inner_dim, 'dropout': config.dropout,
                       'embedding_weights': self.embeddings, 'lexical_switch': args.lexical,
                       'emb_share_prj': args.emb_share_prj, 'global_encoding':config.global_encoding,
                       'encoding_gate': config.encoding_gate, 'stack': config.stack, 'topic':config.topic, 'inception':config.inception,
                       'gtu': config.gtu}
        #if config.multi_gpu:
        #    self.m_args['batch_size'] *= n_gpu
        self.error_file = open(config.error_filename, "w")

    def setup_train(self):
        logging.info('Create model')
        if self.args.load_model is not None:
            with open(os.path.join(config.save_model_path, "model.args"), "r") as f:
                self.m_args = json.loads(f.readline())
                self.m_args['embedding_weights'] = self.embeddings
        self.model = TransformerSummarizer(**self.m_args).to(self.device)
        if config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3])
        # self.model = get_cuda(self.model)
        self.m_args['embedding_weights'] = None
        optim_param = self.model.module.learnable_parameters() if isinstance(self.model, torch.nn.DataParallel) else self.model.learnable_parameters()

        self.optim = Optim(config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
        self.optim.set_parameters(optim_param)
        #self.optim = Adam(optim_param, lr=config.learning_rate, amsgrad=True, betas=[0.9, 0.98], eps=1e-9)
        start_iter = 0

        if self.args.load_model is not None:
            load_model_path = os.path.join(config.save_model_path, self.args.load_model)
            checkpoint = torch.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.module.load_state_dict(checkpoint["model_dict"])
            self.optim = checkpoint['trainer_dict']
            logging.info("Loaded model at " + load_model_path)

        self.print_args()

        return start_iter

    def print_args(self):
        print("encoder layer num: ", config.enc_n_layers)
        print("decoder layer num: ", config.dec_n_layers)
        print("emb_size: ", config.emb_size)
        print("model_dim: ", config.model_dim)
        print("inner_dim: ", config.inner_dim)
        print("mle_weight: ", config.mle_weight)


    def save_model(self, iter):
        save_path = config.dump_filename + "/%07d.tar" % iter
        torch.save({
            "iter": iter + 1,
            "model_dict": self.model.module.state_dict() if config.multi_gpu else self.model.state_dict(),
            "trainer_dict": self.optim
        }, save_path)
        with open(config.args_filename, 'w') as f:
            f.write(json.dumps(self.m_args))
            f.write("\nlearning rate: " + str(config.learning_rate)+"\n")
            f.write("iters_ml/rl: " + str(config.mle_epoch) + "  " + str(config.rl_epoch)+"\n")
            f.write("dump_model_path: " + str(config.dump_filename)+"\n")
            #f.write("topic_flag: "+str(self.args.topic)+"\n")
            f.write("args: "+str(self.args)+"\n")
        logging.info('Model has been saved at %s', save_path)


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
                    score = [{"rouge-1": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-1"]["f"] for score in scores]
        return rouge_l_f1

    def eval_model(self):
        decoded_sents, ref_sents, article_sents = [], [], []
        self.model.module.eval()
        for i in range(0, self.eval_loader.data_length, config.eval_bs):
            start = i
            end = min(i + config.eval_bs, self.eval_loader.data_length)
            batch = self.eval_loader.eval_next_batch(start, end, self.device)
            lengths, indices = torch.sort(batch.src_length, dim=0, descending=True)
            src = torch.index_select(batch.src, dim=0, index=indices)
            trg = torch.index_select(batch.trg, dim=0, index=indices)
            seq, loss = self.model.module.evaluate(src, lengths, max_seq_len=config.max_seq_len, beam_size=None, topic_seq=None, trg=trg)
            try:
                generated = self.loader.decode(seq)
                original = self.loader.decode(trg)
                #original = batch.trg_text
                article = self.loader.decode(src)

                decoded_sents.extend(generated)
                ref_sents.extend(original)
                article_sents.extend(article)
            except:
                print("failed at batch %d", i)
        scores = self.reward_function(decoded_sents, ref_sents)
        score = np.array(scores).mean()
        return score, loss

    def trainIters(self):
        logging.info('Start training')
        start_iter = self.setup_train()
        count = loss_total = rouge_total = 0
        record_rouge = 0.0
        if config.schedule:
            scheduler = L.CosineAnnealingLR(self.optim.optimizer, T_max=config.mle_epoch)

        batches = self.loader.data_length // config.train_bs

        mle_iter = config.mle_epoch * batches
        logging.info("%d steps for per epoch, there is %d epoches.", batches, config.mle_epoch)
        tmp_epoch = (start_iter // batches) + 1
        for i in range(start_iter, mle_iter):
            self.model.train()
            self.model.zero_grad()
            train_batch = self.loader.next_batch(config.train_bs, self.device)
            lengths, indices = torch.sort(train_batch.src_length, dim=0, descending=True)
            src = torch.index_select(train_batch.src, dim=0, index=indices)
            trg = torch.index_select(train_batch.trg, dim=0, index=indices)

            #self.optim.zero_grad()
            if config.topic:
                #print(train_batch.topic)
                loss, seq = self.model.forward(src, trg, lengths, train_batch.topic)
            else:
                loss, seq = self.model.forward(src, trg, lengths)
            if self.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            self.optim.step()

            generated = self.loader.decode(seq)
            original = self.loader.decode(trg)
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

            if i % config.train_sample_interval == 0 and i > 10000:
                score, loss = self.eval_model()
                self.writer.add_scalar('Test', loss, i)
                self.writer.add_scalar('TestRouge', score, i)
                logging.info("%s loss: %f", eval_type, loss)
                if score > record_rouge:
                    logging.info("%s score: %f", eval_type, score)
                    self.save_model(i)
                    record_rouge = score
                elif i % config.save_interval ==0:
                    self.save_model(i)

            if i % batches == 0 and i>1:
                logging.info("Epoch %d finished!", tmp_epoch)
                self.optim.updateLearningRate(score=0, epoch=tmp_epoch)
                if config.schedule:
                    scheduler.step()
                    logging.info("Decaying learning rate to %g" % scheduler.get_lr()[0])
                tmp_epoch = tmp_epoch + 1

        logging.info("mle training finished!")


if __name__ == "__main__":
    description = 'NN model for abstractive summarization based on transformer model.'
    epilog = 'Model training can be performed in two modes: with pretrained embeddings or train embeddings with model ' \
             'simultaneously. To choose mode use --pretrain_emb argument. ' \
             'Please, use deeper model configuration than this, if you want obtain good results.'

    parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--lexical', action='store_true', help='whether to use lexical shortcut')
    parser.add_argument('--fine_tune', action='store_true', help='whether to use fine tune transformer')
    parser.add_argument('--pretrain_emb', action='store_true', help='use pretrained embeddings')
    parser.add_argument('--emb_share_prj', action='store_true', help='share embeddings weight with target proj')
    parser.add_argument("--load_model", type=str, default=None)

    args = parser.parse_args()

    print(config.this_model_path)
    logging.info('saved at %s', config.this_model_path)

    print_config = {}
    opt = vars(args)
    for key in opt:
        if key not in print_config:
            print_config[key] = opt[key]
    for k, v in print_config.items():
        logging.info("%s:\t%s\n" % (str(k), str(v)))

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
