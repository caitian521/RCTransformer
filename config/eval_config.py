import datetime
import os

dataset = '/data/ct/abs_summarize/sumdata/transf_summ/dataset'
prefix = 'lexical-summ'



'''-----------------CNNDailyMail-------------------'''
'''
dataset = "/data/ct/abs_summarize/cnndm-pj/dataset"
prefix = '/cnndm'
'''
model_path = os.path.join(dataset, "../final_dumps")
corpus_model_path = os.path.join(dataset,prefix)
#corpus_model_path = os.path.join(dataset,"vocab_model_dumps")
log = "./logs"

timestamp = "04-20-08"
#timestamp = "04-17-15"
this_model_path =os.path.abspath(os.path.join(model_path,timestamp))

vocab_size = 15000 #giga25000 cnn30000
emb_size = 250 #giga250 cnn300
model_dim = 512

inner_dim = 512
dropout = 0.1
learning_rate = 0.00025
iters = 75000   #30000
rl_iters = 90000  #60000
test_bs = 12
multi_gpu = True


max_seq_len = 20


train_interval = 100
train_sample_interval = 100
save_interval = 1000

test_interval = 100
sample_interval = 200


error ='/error'
model_name = "/model"

emb_filename = corpus_model_path + '/embedding.npy'
bpe_model_filename = corpus_model_path + prefix + '_bpe.model'

log_filename = this_model_path + prefix + '.log'
error_filename = this_model_path + error + '.txt'
dump_filename = this_model_path
args_filename = this_model_path + model_name + '.args'

save_model_path = this_model_path
#save_model_path = "/data/ct/abs_summarize/sumdata/transf_summ/model_dumps/03-21-09" #03-20-10"






