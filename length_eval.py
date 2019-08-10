import os
from config import eval_config
import codecs
import logging
import pyrouge
docs = "/data/ct/text-summarization-tensorflow/metric_doc"

result = "/data/ct/text-summarization-tensorflow/result.txt"
article = "/data/ct/text-summarization-tensorflow/sumdata/test/valid.article.filter.txt"
references = "/data/ct/text-summarization-tensorflow/sumdata/test/valid.title.filter.txt"



def print_for_rouge(decoded_sents, ref_sents, corpus="giga", lenth=None):
    assert len(decoded_sents) == len(ref_sents)
    if lenth is not None:
        ref_dir = os.path.join(docs, 'reference+%s' % lenth)
        cand_dir = os.path.join(docs, 'candidate+%s' % lenth)
    else:
        ref_dir = os.path.join(docs, 'reference')
        cand_dir = os.path.join(docs, 'candidate')
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    print(ref_dir)
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
    #eval_file.write("F_measure: %s Recall: %s Precision: %s\n" % (str(f_score), str(recall), str(precision)))

ref = open(references).readlines()
arc = open(article).readlines()

d,r = [],[]
with open(result) as res:
    for i,line in enumerate(res):
            d.append(line.strip())
            r.append(ref[i].strip())
print_for_rouge(d, r, corpus="giga", lenth=None)