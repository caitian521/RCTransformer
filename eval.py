import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import logging
'''
def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
'''

data_type = "val"
duc_src_type = "input"
duc_name = 'DUC2004'
directory = os.path.join(eval_config.dataset, duc_name)


class Evaluate(object):
    def __init__(self, args):
        #self.loader = DataLoader(eval_config.dataset, data_type, ['src', 'trg'], eval_config.bpe_model_filename, eval_config.vocab_size)
        self.loader = EvalBatcher(eval_config.dataset, duc_name, duc_src_type, eval_config.bpe_model_filename)

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


    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = data_type + "_" + loadfile.split(".")[0] + ".txt"

        with open(os.path.join(eval_config.save_model_path, filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i][0].strip().lower() + "\n")
                f.write("ref: " + ref_sents[i][1].strip().lower() + "\n")
                f.write("ref: " + ref_sents[i][2].strip().lower() + "\n")
                f.write("ref: " + ref_sents[i][3].strip().lower() + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def print_for_rouge(self,decoded_sents, ref_sents, corpus="giga"):
        assert len(decoded_sents) == len(ref_sents)
        ref_dir = os.path.join(eval_config.save_model_path, 'reference')
        cand_dir = os.path.join(eval_config.save_model_path, 'candidate')
        if not os.path.exists(ref_dir):
            os.mkdir(ref_dir)
        if not os.path.exists(cand_dir):
            os.mkdir(cand_dir)
        if corpus == "giga":
            for i in range(len(ref_sents)):
                with codecs.open(ref_dir + "/%06d_reference.txt" % i, 'w+', 'utf-8') as f:
                    f.write(ref_sents[i])
                with codecs.open(cand_dir + "/%06d_candidate.txt" % i, 'w+', 'utf-8') as f:
                    f.write(decoded_sents[i])
            r = pyrouge.Rouge155()
            r.model_filename_pattern = '#ID#_reference.txt'
            r.system_filename_pattern = '(\d+)_candidate.txt'
        else:
            for i in range(len(ref_sents)):
                nickname = ['A', 'B', 'C', 'D']
                for task in range(len(ref_sents[0])):
                    ref_file_name = nickname[task] + ".%06d_reference.txt" % i
                    with codecs.open(os.path.join(ref_dir, ref_file_name), 'w+', 'utf-8') as f:
                        f.write(ref_sents[i][task].strip().lower())
                with codecs.open(cand_dir + "/%06d_candidate.txt" % i, 'w+', 'utf-8') as f:
                    f.write(decoded_sents[i])
            r = pyrouge.Rouge155()
            r.model_filename_pattern = '[A-Z].#ID#_reference.txt'
            r.system_filename_pattern = '(\d+)_candidate.txt'


        r.model_dir = ref_dir
        r.system_dir = cand_dir
        logging.getLogger('global').setLevel(logging.WARNING)
        rouge_results = r.convert_and_evaluate()
        scores = r.output_to_dict(rouge_results)
        recall = [round(scores["rouge_1_recall"] * 100, 2),
                  round(scores["rouge_2_recall"] * 100, 2),
                  round(scores["rouge_l_recall"] * 100, 2)]
        precision = [round(scores["rouge_1_precision"] * 100, 2),
                     round(scores["rouge_2_precision"] * 100, 2),
                     round(scores["rouge_l_precision"] * 100, 2)]
        f_score = [round(scores["rouge_1_f_score"] * 100, 2),
                   round(scores["rouge_2_f_score"] * 100, 2),
                   round(scores["rouge_l_f_score"] * 100, 2)]
        print("F_measure: %s Recall: %s Precision: %s\n" % (str(f_score), str(recall), str(precision)))


    def cal_rouge(self, decoded_sents, original_sents):
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
                    score = [{"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [[score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]] for score in scores]
        return rouge_l_f1


    def evaluate_batch(self, max_seq_len, beam_size=None, corpus_id=-1):
        decoded_sents = []
        ref_sents = []
        article_sents = []
        for i in range(0, self.loader.data_length, eval_config.test_bs):
            start = i
            end = min(i + eval_config.test_bs, self.loader.data_length)
            batch = self.loader.eval_next_batch(start, end, self.device)
            lengths, indices = torch.sort(batch.src_length, dim=0, descending=True)
            src = torch.index_select(batch.src, dim=0, index=indices)

            topic_seq = None
            seq, loss = self.model.evaluate(src, lengths, max_seq_len=max_seq_len, beam_size=beam_size, topic_seq=topic_seq)
            # print("success", i)
            with torch.autograd.no_grad():
                generated = self.loader.decode_eval(seq)
                article = self.loader.decode_eval(src)
                if corpus_id == -1:
                    trg = torch.index_select(batch.trg, dim=0, index=indices)
                    original = self.loader.decode_eval(trg)
                    ref_sents.extend(original)
                else:
                    ref_sents.extend((indices.cpu().numpy()+i).tolist())

                decoded_sents.extend(generated)
                article_sents.extend(article)

        return decoded_sents,ref_sents, article_sents

    def run(self, max_seq_len, beam_size=None, print_sents=False, corpus_name="giga"):
        self.setup_valid()

        load_file = self.args.load_model
        if corpus_name=='duc':
            task_files = ['task1_ref0.txt', 'task1_ref1.txt', 'task1_ref2.txt', 'task1_ref3.txt']
            self.duc_target = EvalTarget(directory, task_files)
            decoded_sents, ref_id, article_sents = self.evaluate_batch(max_seq_len, beam_size=beam_size, corpus_id=1)
            ref_sents = []

            for id in ref_id:
                trg_sents = []
                for i in range(len(task_files)):
                    trg_sents.append(self.duc_target.text[i][id])

                ref_sents.append(trg_sents)
            '''
            rouge = Rouge()
            max_scores = []
            ref_for_print = []
            for mine, labels in zip(decoded_sents, ref_sents):
                scores = []
                for sentence in labels:
                    scores.append(rouge.get_scores(mine, sentence, avg=True)['rouge-1']['f']) #
                maxx = max(scores)
                max_scores.append(maxx)
                ref_for_print.append(scores.index(maxx))
            #print(max_scores)
            all_score = np.array(max_scores).mean(axis=0)
            #print(all_score)
            '''
            if print_sents:
                self.print_for_rouge(decoded_sents, ref_sents, corpus="duc")
                #print(np.array(ref_sents).size)
                self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)

        else:
            decoded_sents, ref_sents, article_sents = self.evaluate_batch(max_seq_len, beam_size=beam_size, corpus_id=-1)
            scores = self.cal_rouge(decoded_sents, ref_sents)
            score = np.array(scores).mean(axis=0)
            print("rouge-1: %.4f, rouge-2: %.4f, rouge-l: %.4f" % (score[0], score[1], score[2]))

            if print_sents:
                self.print_for_rouge(decoded_sents, ref_sents)
                #self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate", "test"])
    parser.add_argument("--start_from", type=str, default="0200000.tar")
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
        for f in saved_models:
            if f.find("tar")!=-1:
                print(f)
                eval_processor.args.load_model = f
                eval_processor.run(max_seq_len=20, beam_size=5,print_sents=True,corpus_name="duc")
                #eval_processor.run(max_seq_len=eval_config.max_seq_len, beam_size=5,print_sents=True,corpus_name="giga")
    else:  # test
        eval_processor = Evaluate(args)
        eval_processor.run(beam_size=5)
