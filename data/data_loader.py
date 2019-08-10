import csv
import os

import numpy as np
import sentencepiece as spm
import torch
from data.generate_topic import GenerateTopic
from config import config

class DataLoader:
    def __init__(self, directory, part, cols, spm_filename, vocab_size=15000, topic = false):
        """Dataset loader.
        Args:
            directory (str): dataset directory.
            parts (list[str]): dataset parts. [parts].tsv files must exists in dataset directory.
            spm_filename (str): file name of the dump sentencepiece model.
        """
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)
        self.vocab_size = vocab_size
        #self.mask_idx = 30000 #词表最后一行，训练摘要模型的时候需删掉
        self.cols = cols
        self.inputfile = os.path.join(directory, part)
        self.spm_filename = spm_filename
        self.topic = topic
        if topic:
            self.topic_maker = GenerateTopic(directory, part)
            self.topic = self.topic_maker.read_topic()

        # Load sentecepiece model:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spm_filename)

        # Load dataset parts:
        self.data = list(self.from_tsv())
        self.data_length = len(self.data)
        self.max_len = self.get_max_len()


    def next_batch(self, batch_size, device):
        """Get next batch.
        Args:
            batch_size (int): batch size.
            part (str): dataset part.
            device (torch.device): torch device.
        Returns:
            Batch: batch wrapper.
        """
        indexes = np.random.randint(0, self.data_length, batch_size)
        src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source = [],[],[],[],[],[],[]
        topic = []
        for i in indexes:
            src_input.append(self.data[i][0])
            trg_input.append(self.data[i][1])
            src_extend_ids.append(self.data[i][2])
            src_oov.append(self.data[i][3])
            trg_extend_ids.append(self.data[i][4])
            trg_text.append(self.data[i][5])

            source.append(self.data[i][6])
            if self.topic:
                topic.append(self.sp.EncodeAsIds(self.topic[i]))
        #print(topic)
        if self.topic:
            return Batch(self,  src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source, device, topic)
        else:
            return Batch(self, src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source, device)


    def eval_next_batch(self,start, end, device):
        indexes = np.arange(start, end)
        '''
        raw_batches = [[self.data[i][col] for i in indexes] for col, name in enumerate(self.cols)]
        topic_batch = [self.sp.EncodeAsIds(self.topic[i]) for i in indexes]
        return Batch(self, raw_batches, device, topic_batch)
        '''
        src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, topic, source = [], [], [], [], [], [], [], []
        for i in indexes:
            src_input.append(self.data[i][0])
            trg_input.append(self.data[i][1])
            src_extend_ids.append(self.data[i][2])
            src_oov.append(self.data[i][3])
            trg_extend_ids.append(self.data[i][4])
            trg_text.append(self.data[i][5])
            source.append(self.data[i][6])
            if self.topic:
                topic.append(self.sp.EncodeAsIds(self.topic[i]))
        if self.topic:
            return Batch(self, src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source, device, topic)
        else:
            return Batch(self, src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source, device)

    def sequential(self, device):
        """Get all examples from dataset sequential.
        Args:
            part (str): part of the dataset.
            device: (torch.Device): torch device.
        Returns:
            Batch: batch wrapper with size 1.
        """
        for example in self.data:
            raw_batches = [example]
            yield Batch(self, raw_batches, device)

    def pad(self, data, topic_flag=False):
        """Add <sos>, <eos> tags and pad sequences from batch
        Args:
           data (list[list[int]]): token indexes
        Returns:
            list[list[int]]: padded list of sizes (batch, max_seq_len + 2)
        """
        if topic_flag == False:
            data = list(map(lambda x: [self.sos_idx] + x + [self.eos_idx], data))
        else:
            data = list(data)
        lens = [len(s) for s in data]
        max_len = max(lens)
        for i, length in enumerate(lens):
            to_add = max_len - len(data[i])
            data[i] += [self.pad_idx] * to_add
        return data, lens


    def from_tsv(self, src_trunc_len=-1):
        """Read and tokenize data from TSV file.
            Args:
                part (str): the name of the part.
            Yields:
                (list[int], list[int]): pairs for each example in dataset.
        """
        with open(self.inputfile+".tsv") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                source = row[0]
                source_ids = self.sp.EncodeAsIds(row[0])
                source_words = self.sp.EncodeAsPieces(row[0])
                source_extend_ids, source_oov = self.src_extend(source_ids[:src_trunc_len], source_words[:src_trunc_len])
                target_ids = self.sp.EncodeAsIds(row[1])
                target_text = row[1]
                target_extend_ids = self.trg_extend(target_ids, source_oov)
                yield tuple([source_ids, target_ids, source_extend_ids, source_oov, target_extend_ids, target_text, source])

    def src_extend(self, source_ids, source_words):
        oovs = []
        ids = []
        for i, item in enumerate(source_ids):
            if item == self.unk_idx:
                # Add to list of OOVs
                w = source_words[i]
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(self.vocab_size + oov_num)
            else:
                ids.append(item)
        return ids, oovs

    def trg_extend(self, target_ids, source_oov):
        ids = []
        unk_id = self.unk_idx
        for w in target_ids:
            if w == unk_id:  # If w is an OOV word
                if w in source_oov:  # If w is an in-article OOV
                    vocab_idx = self.vocab_size + source_oov.index(w)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(unk_id)  # Map to the UNK token id
            else:
                ids.append(w)
        return ids

    def decode(self, data):
        """Decode encoded sentence tensor.
        Args:
            data (torch.Tensor): sentence tensor.
        Returns:
            list[str]: decoded sentences.
        """
        result = []
        for sentence in data:
            token_list = []
            for token in sentence:
                if type(data) is torch.Tensor:
                    if token.item() == self.eos_idx or token.item() == self.pad_idx:
                        break
                    token_list.append(token.item())
                else:
                    if token == self.eos_idx or token == self.pad_idx:
                        break
                    token_list.append(token)
            result.append(self.sp.DecodeIds(token_list))
        return result

    def decode_oov(self, data, source_oovs, oov):
        """Decode encoded sentence tensor.
        Args:
            data (torch.Tensor): sentence tensor.
        Returns:
            list[str]: decoded sentences.
        """
        result = []
        cnt = 0
        for sentence in data:
            token_list = []
            for i, token in enumerate(sentence):
                if type(data) is torch.Tensor:
                    if token.item() == self.eos_idx or token.item() == self.pad_idx:
                        break
                    token_list.append(token.item())
                else:
                    if token == self.eos_idx or token == self.pad_idx:
                        break
                    token_list.append(token)

            decoded_seq = self.sp.DecodeIds(token_list)
            new_token_list = decoded_seq.split(" ")
            max_len = len(new_token_list)

            for i, item in enumerate(oov[cnt]):
                if i+1 >= max_len:
                    break
                if oov[cnt][i] > self.vocab_size:
                    try:
                        oov_word = source_oovs[cnt][oov[cnt][i] - self.vocab_size]
                        print(oov_word)
                        new_token_list[i+1] = oov_word
                    except:  # i doesn't correspond to an article oov
                        print('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                            oov[cnt][i], cnt, len(source_oovs[cnt])))
            decoded_seq = " ".join([item for item in new_token_list])
            #print(decoded_seq)
            result.append(decoded_seq)
            cnt = cnt+1
        return result


    def decode_eval(self, data):
        if type(data) is torch.Tensor:
            return [self.sp.DecodeIds([token.item() for token in sentence]) for sentence in data]
        else:
            return [self.sp.DecodeIds([token for token in sentence]) for sentence in data]

    def decode_raw(self, data):
        """Decode encoded sentence tensor without removing auxiliary symbols.
                Args:
                    data (torch.Tensor): sentence tensor.
                Returns:
                    list[str]: decoded sentences.
                """
        return [''.join([self.sp.IdToPiece(token.item()) for token in sentence]) for sentence in data]

    def get_max_len(self):
        lens = []
        for example in self.data:
            for col in example:
                lens.append(len(col))
        return max(lens) + 2


class Batch:
    def __init__(self, data_loader, src_input, trg_input, src_extend_ids, src_oov, trg_extend_ids, trg_text, source, device, topic=None):
        """Simple batch wrapper.
        Args:
            data_loader (DataLoader): data loader object.
            raw_batches (list[data]): raw data batches.
            device (torch.device): torch device.
        Variables:
            - **cols_name_length** (list[int]): lengths of `cols_name` sequences.
            - **cols_name** (torch.Tensor): long tensor of `cols_name` sequences.
        """
        #print("make batch")
        #print(device)

        tensor, length = data_loader.pad(src_input)
        max_src_seq_len = max(length)
        self.__setattr__("src", torch.tensor(tensor, dtype=torch.long, device=device))
        self.__setattr__("src" + '_length', torch.tensor(length, dtype=torch.long, device=device))

        tensor, length = data_loader.pad(trg_input)
        max_trg_seq_len = max(length)
        self.__setattr__("trg", torch.tensor(tensor, dtype=torch.long, device=device))
        self.__setattr__("trg" + '_length', length)

        max_oov_len = max([len(oov) for oov in src_oov])
        self.__setattr__("source_oov", src_oov)
        batch_size = len(src_input)
        extra_zeros = np.zeros((batch_size, max_oov_len), dtype=np.float)
        if max_oov_len > 0:
            self.__setattr__("extra_zeros", torch.tensor(extra_zeros, dtype=torch.float, device=device))
        else:
            self.__setattr__("extra_zeros", None)

        extend_ids = np.zeros((batch_size, max_src_seq_len), dtype=np.int32)
        for i in range(batch_size):
            extend_ids[i, :len(src_extend_ids[i])] = src_extend_ids[i][:]
        self.__setattr__("enc_batch_extend_vocab", torch.tensor(extend_ids, dtype=torch.long, device=device))

        trg_extend = np.zeros((batch_size, max_trg_seq_len), dtype=np.int32)
        for i in range(batch_size):
            trg_extend[i, :len(trg_extend_ids[i])] = trg_extend_ids[i]
        self.__setattr__("trg_extend_vocab", torch.tensor(trg_extend, dtype=torch.long, device=device))

        self.__setattr__("trg_text", trg_text)
        self.__setattr__("src_text", source)

        if topic is not None:
            tensor_topic, topic_length = data_loader.pad(topic, topic_flag=True)
            self.__setattr__("topic", torch.tensor(tensor_topic, dtype=torch.long, device=device))


class CNNDailyMail:
    def __init__(self, directory, part, cols, spm_filename):
        """Dataset loader.
        Args:
            directory (str): dataset directory.
            parts (list[str]): dataset parts. [parts].tsv files must exists in dataset directory.
            spm_filename (str): file name of the dump sentencepiece model.
        """
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)
        self.cols = cols
        self.input_file = os.path.join(directory, part)


        # Load sentecepiece model:
        self.spm_filename = spm_filename
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spm_filename)

        # Load dataset parts:
        self.data = list(self.from_file())
        self.data_length = len(self.data)
        self.max_len = self.get_max_len()

    def next_batch(self, batch_size, device):
        """Get next batch.
        Args:
            batch_size (int): batch size.
            part (str): dataset part.
            device (torch.device): torch device.
        Returns:
            Batch: batch wrapper.
        """
        indexes = np.random.randint(0, self.data_length, batch_size)
        raw_batches = [[self.data[i][col] for i in indexes] for col, name in enumerate(self.cols)]
        #topic_batch = [self.sp.EncodeAsIds(self.topic[i]) for i in indexes]
        return Batch(self, raw_batches, device)#, topic_batch)

    def eval_next_batch(self,start, end, device):
        indexes = np.arange(start, end)
        raw_batches = [[self.data[i][col] for i in indexes] for col, name in enumerate(self.cols)]
        topic_batch = [self.sp.EncodeAsIds(self.topic[i]) for i in indexes]
        return Batch(self, raw_batches, device, topic_batch)


    def pad(self, data, topic_flag=False):
        """Add <sos>, <eos> tags and pad sequences from batch
        Args:
           data (list[list[int]]): token indexes
        Returns:
            list[list[int]]: padded list of sizes (batch, max_seq_len + 2)
        """
        if topic_flag == False:
            data = list(map(lambda x: [self.sos_idx] + x + [self.eos_idx], data))
        else:
            data = list(data)
        lens = [len(s) for s in data]
        max_len = max(lens)
        max_len = min(max_len, 400)
        for i, length in enumerate(lens):
            if length <= max_len:
                to_add = max_len - length
                data[i] += [self.pad_idx] * to_add
            else:
                data[i] = data[i][:max_len]
                lens[i] = max_len
        return data, lens



    def from_file(self, encoding="utf-8", quotechar=None):
        """Reads a tab separated value file."""

        def get_abstract(abstract):
            cur = 0
            sents = []
            while True:
                try:
                    start_p = abstract.index("<s>", cur)
                    end_p = abstract.index("</s>", start_p + 1)
                    cur = end_p + len("</s>")
                    sents.append(abstract[start_p + len("<s>"):end_p])
                except ValueError as e:  # no more sentences
                    break

            line = ' '.join(sents)
            return line

        with open(self.input_file+".txt", "r", encoding=encoding) as f:
            reader = csv.reader(f, delimiter="\n", quotechar=quotechar)
            for line in reader:
                #print(line)
                line = line[0].split("<summ-content>")
                if len(line)==2:
                    src_seq = line[1]
                    tgt_seq = get_abstract(line[0])
                    yield (self.sp.EncodeAsIds(src_seq), self.sp.EncodeAsIds(tgt_seq))
                else:
                    print("ERROR:" + line)
                    continue

    def decode(self, data):
        """Decode encoded sentence tensor.
        Args:
            data (torch.Tensor): sentence tensor.
        Returns:
            list[str]: decoded sentences.
        """
        if type(data) is torch.Tensor:
            return [self.sp.DecodeIds([token.item() for token in sentence]) for sentence in data]
        else:
            return [self.sp.DecodeIds([token for token in sentence]) for sentence in data]


    def decode_raw(self, data):
        """Decode encoded sentence tensor without removing auxiliary symbols.
                Args:
                    data (torch.Tensor): sentence tensor.
                Returns:
                    list[str]: decoded sentences.
                """
        return [''.join([self.sp.IdToPiece(token.item()) for token in sentence]) for sentence in data]

    def get_max_len(self):
        lens = []
        for example in self.data:
            for col in example:
                lens.append(len(col))
        return max(lens) + 2

if __name__ == "__main__":
    #topic_maker = GenerateTopic("/data/ct/abs_summarize/sumdata/transf_summ/dataset", "train_small")
    #topic = topic_maker.read_topic()

    bpe_path = config.bpe_model_filename
    loader = DataLoader(config.dataset, "train_small", ["src","trg"],bpe_path)
    device = torch.device("cuda")
    batch = loader.next_batch(32, device)
    lengths, indices = torch.sort(batch.src_length, dim=0, descending=True)
    print(indices.cpu().numpy())
    src = torch.index_select(batch.src, dim=0, index=indices)
    tgt = torch.index_select(batch.trg, dim=0, index=indices)
    print(indices)
    print(src)
    print(tgt)
    '''
    bpe_path = config.bpe_model_filename
    dataset = config.dataset
    part = "train_small"
    loader = CNNDailyMail(dataset, part, ["src","trg"], bpe_path)
    device = torch.device("cuda")
    batch = loader.next_batch(32, device)
    print(loader.data_length)
    print(loader.max_len)
    print(batch.trg)
    '''