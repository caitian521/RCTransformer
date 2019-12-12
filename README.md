# RC-Transformer
## Improving Transformer with Sequential Context Representations for Abstractive Text Summarization
 Recent dominant approaches for abstractive text summarization are mainly RNN-based encoder-decoder framework, these methods usually suffer from the poor semantic representations for long sequences. In this paper, we propose a new abstractive summarization model, called RC-Transformer (RCT). The model is not only capable of learning long-term dependencies, but also addresses the inherent shortcoming of Transformer on insensitivity to word order information.  
 
 We extend the Transformer with an additional RNN-based encoder to capture the sequential context representations. In order to extract salient information effectively, we further construct a convolution module to filter the sequential context with local importance. The experimental results on Gigaword and DUC-2004 datasets show that our proposed model achieves the state-of-the-art performance, even without introducing external information. In addition, our model also owns an advantage in speed over the RNN-based models. 

## Overview 

[Image text](https://github.com/caitian521/RCTransformer/blob/master/picture/overview.png)

## Training
If you want to train your own model, please follow the following steps. Two GPUs with 12GB memory or more will be helpful.  

1. Prepare a parallel linguistics corpus for abstractive text summarization (without tokenization), like [Gigaword](https://github.com/Ethanscuter/gigaword) or [CNN Daily/Mail](https://github.com/abisee/cnn-dailymail).  
2. Create a dataset folder. Set the prefix, vocab_size, emb_size in config/config.py.  
3. Run python preprocess.py to generate sub-word vocabulary and word2vec embeddings. The input format should be "source"\t"summary". The *_bpe.vocab and word2vbvec.model will appear in the dataset/prefix folder.  
4. Edit the config/config.py for training, Run python train_run.py --cuda --lexical --pretrain_emb. It will take at least 2 days to train a good model with batch_size 64. 
5. We also implement a REINFORCE training, running python train_RL.py --cuda --lexical --pretrain_emb after step4 always get a better model.  
6. *Copy mechanism is also introduced in our experiments. But we did not achieve the desired results. Maybe effective for the CNN Daily/Mail dataset  

## Testing  
1. please set configuration in config.py
2. run
 ```bash
 python eval.py --cuda --lexical --pretrain_emb
 ```

## Results

| | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---- | ---- | ---- | ---- |
| Gigaword | 37.16	| 17.73	|34.41|
| DUC2004| 33.16	| 14.7|	30.52|

[details see paper](https://github.com/caitian521/RCTransformer/blob/master/picture/Improving%20Transformer%20with%20Sequential%20Context%20Representations%20for%20Abstractive%20Text%20Summarization.pdf)


## Other


```
.
├── config             配置文件
│   ├── config.py      模型训练
│   └── eval_config.py 模型预测
├── data
│   ├── data_loader.py load训练数据
│   ├── eval_batcher.py load测试数据
│   ├── generate_topic.py 生成keyword
│   ├── __init__.py
│   └── utils.py   dataloader工具
├── eval.py            模型预测
├── finetune.py        用语言模型微调
├── length_eval.py     实验：不同输出文本长度的影响
├── model
│   ├── beam_search.py  beam search
│   ├── __init__.py
│   ├── lr_scheduler.py 学习率衰减
│   ├── model.py        模型
│   └── optims.py       optimizer优化
├── nn
│   ├── __init__.py
│   ├── modules
│   │   ├── attention.py      self-attention
│   │   ├── embedding.py         通用embedding
│   │   ├── __init__.py
│   │   ├── layers.py            encoder decoder layer
│   │   ├── position_wise.py     position_wise feed forward layer
│   └── └── transformer.py       label smoothing
├── preprocess.py                数据预处理，分词，训练embedding
├── README.md
├── requirements.txt
├── train_RL.py                强化训练
├── train_run.py               模型训练
├── train_copy.py              copy机制
└── visual_attention.py        attention可视化
```
