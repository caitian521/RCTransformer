import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from rouge import Rouge
import time
from data import DataLoader
from data import EvalTarget, EvalBatcher
from model import TransformerSummarizer
from config import eval_config
import codecs
import json
import pyrouge
import matplotlib.pyplot as plt
import logging
'''
def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
'''
data_type = "sample"
duc_src_type = "input"
duc_name = 'DUC2004'
directory = os.path.join(eval_config.dataset, duc_name)


class Evaluate(object):
    def __init__(self, args):
        self.loader = DataLoader(eval_config.dataset, data_type, ['src', 'trg'], eval_config.bpe_model_filename, eval_config.vocab_size)
        #self.loader = EvalBatcher(eval_config.dataset, duc_name, duc_src_type, eval_config.bpe_model_filename)

        if args.pretrain_emb:
            self.embeddings = torch.from_numpy(np.load(eval_config.emb_filename)).float()
            self.vocab_size, self.emb_size = self.embeddings.shape
        else:
            self.embeddings = None
            self.vocab_size, self.emb_size = eval_config.vocab_size, eval_config.emb_size
        with open(eval_config.args_filename, "r") as f:
            self.m_args = json.loads(f.readline())
            #self.m_args['max_seq_len'] = eval_config.max_seq_len

        time.sleep(5)
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        #self.iters = int(self.loader.data_length / self.m_args['batch_size'])

    def setup_valid(self):
        if self.args.cuda:
            checkpoint = torch.load(os.path.join(eval_config.save_model_path, self.args.load_model))
        else:
            checkpoint = torch.load(os.path.join(eval_config.save_model_path, self.args.load_model), map_location = lambda storage, loc: storage)
        if self.args.fine_tune:
            embeddings = checkpoint["model_dict"]["positional_encoding.embedding.weight"]
            self.m_args["embedding_weights"] = embeddings
        #self.m_args['embedding_weights'] = self.embeddings
        self.model = TransformerSummarizer(**self.m_args).to(self.device)
        self.model.load_state_dict(checkpoint["model_dict"])





    def evaluate_batch(self):
        self.setup_valid()
        decoded_sents = []
        ref_sents = []
        article_sents = []
        for i in range(0, self.loader.data_length, eval_config.sample_bs):
            start = i
            end = min(i + eval_config.test_bs, self.loader.data_length)
            batch = self.loader.eval_next_batch(start, end, self.device)
            lengths, indices = torch.sort(batch.src_length, dim=0, descending=True)
            src = torch.index_select(batch.src, dim=0, index=indices)
            trg = torch.index_select(batch.trg, dim=0, index=indices)
            topic_seq = None

            loss, seq, att_en, att_rc = self.model.forward(src, trg, lengths)
            # print("success", i)
            with torch.autograd.no_grad():
                generated = self.loader.decode_raw(seq)
                article = self.loader.decode_raw(src)

                ref_sents.extend((indices.cpu().numpy()+i).tolist())

                decoded_sents.extend(generated)
                article_sents.extend(article)
        #print(att_rc.shape)
        #print(att_en.shape)
        #print(generated)
        #return decoded_sents,ref_sents, article_sents
        for i in range(8):
            #plt.figure(i)
            plot_heatmap(article[0], generated[0], att_en[i].cpu().data)

        for i in range(8):
            #plt.figure(i)
            plot_heatmap(article[0], generated[0], att_rc[i].cpu().data)

def plot_heatmap(v, q, scores):
    src = q
    trg = v
    print(src)
    print(trg)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    ax.set_xticklabels(trg, minor=False, rotation='vertical')
    ax.set_yticklabels(src, minor=False)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate", "test"])
    parser.add_argument("--start_from", type=str, default="0155000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--fine_tune', action='store_true', help='whether to use fine tune transformer')
    parser.add_argument('--pretrain_emb', action='store_true', help='use pretrained embeddings')
    parser.add_argument('--topic', action='store_true', help='whether to use topic attention')
    parser.add_argument('--emb_share_prj', action='store_true', help='share embeddings weight with target proj')
    parser.add_argument('--lexical', action='store_true', help='whether to use lexical shortcut')
    args = parser.parse_args()

    if args.task == "validate":
        saved_models = os.listdir(eval_config.save_model_path)
        print("now eval model at : ", eval_config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(args.start_from)
        saved_models = saved_models[file_idx:]
        eval_processor = Evaluate(args)
        if file_idx:
                eval_processor.args.load_model = args.start_from
                #eval_processor.run(max_seq_len=18, beam_size=5,print_sents=True,corpus_name="duc")
                eval_processor.evaluate_batch()
    else:  # test
        eval_processor = Evaluate(args)
        eval_processor.run(beam_size=5)
