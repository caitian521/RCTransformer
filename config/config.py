import datetime
import os

dataset = '/data/ct/abs_summarize/sumdata/transf_summ/dataset'
prefix = 'lexical-summ'
model_name = "test_dumps"


'''-----------------CNNDailyMail-------------------'''
'''
dataset = "/data/ct/abs_summarize/cnndm-pj/dataset"
prefix = '/cnndm'
'''
model_path = os.path.join(dataset, ".." , model_name)
corpus_model_path = os.path.join(dataset,prefix)
#corpus_model_path = os.path.join(dataset,"vocab_model_dumps")


timestamp = datetime.datetime.now().strftime("%m-%d-%H")
this_model_path =os.path.abspath(os.path.join(model_path,timestamp))

vocab_size = 15000 #giga25000 cnn30000
emb_size = 250 #giga250 cnn300

#---------training----------
model_dim = 512
enc_n_layers = 4 #Transformer encoder layer
dec_n_layers = 4 #Transformer decoder layer
n_heads = 8
inner_dim = 1024  # Transformer FFN size

global_layers = 2 # RNN encoder layer
global_encoding = True #use convolutional module to capture local relation
encoding_gate = True #use glu to filter sequential context representation in terms to local relation
inception = True #use a
gtu = False

stack = False #decoder stack or gated sum two encoder-decoder attention

topic=False # transformer encoder introduce topic information


mle_epoch = 10
rl_epoch = 2
train_bs = 96
multi_gpu = True


max_grad_norm = 10
learning_rate_decay = 0.5
start_decay_at = 1
mle_weight = 0.75#强化学习使用
dropout = 0.1
learning_rate = 0.0001
schedule = True
#--------training evaluation-----------
train_interval = 100
train_sample_interval = 1000
save_interval = 2000

test_interval = 100
sample_interval = 500
max_seq_len = 15
eval_bs = 32



emb_filename = os.path.join(corpus_model_path, 'embedding.npy')
bpe_model_filename = os.path.join(corpus_model_path, prefix) + '_bpe.model'

log_filename = os.path.join(this_model_path, prefix) + '.log'
error_filename = os.path.join(this_model_path, "error.txt")
dump_filename = this_model_path
args_filename = os.path.join(this_model_path, "model.args")


#save_model_path = "/data/ct/abs_summarize/sumdata/transf_summ/model_dumps/04-06-23"
save_model_path = "/data/ct/abs_summarize/sumdata/transf_summ/final_dumps/04-16-11" #03-20-10"



