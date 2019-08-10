import csv
import os

import numpy as np
import sentencepiece as spm
import torch
from data.generate_topic import GenerateTopic
from config import config

class EvalTarget:
    def __init__(self, directory, files):
        self.text = []
        for i,file in enumerate(files):
            filename = os.path.join(directory, file)
            self.text.append(self.from_file(filename))

    def from_file(self, filename):
        with open(filename) as f:
            return f.readlines()



class EvalBatcher:
    def __init__(self, directory, name, part, spm_filename, vocab_size=15000):
        """Dataset loader.
        Args:
            directory (str): dataset directory.
            parts (list[str]): dataset parts. [parts].tsv files must exists in dataset directory.
            spm_filename (str): file name of the dump sentencepiece model.
        """
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)
        self.vocab_size = vocab_size
        #self.mask_idx = 30000 #词表最后一行，训练摘要模型的时候需删掉
        self.inputfile = os.path.join(directory, name, part)
        self.spm_filename = spm_filename

        # Load sentecepiece model:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spm_filename)

        # Load dataset parts:
        self.data = list(self.from_txt())
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
        src_input, src_extend_ids, src_oov, source = [],[],[],[]
        for i in indexes:
            src_input.append(self.data[i][0])
            src_extend_ids.append(self.data[i][1])
            src_oov.append(self.data[i][2])
            source.append(self.data[i][3])
        return Batch(self,  src_input, src_extend_ids, src_oov, source, device)

    def eval_next_batch(self,start, end, device):
        indexes = np.arange(start, end)
        '''
        raw_batches = [[self.data[i][col] for i in indexes] for col, name in enumerate(self.cols)]
        topic_batch = [self.sp.EncodeAsIds(self.topic[i]) for i in indexes]
        return Batch(self, raw_batches, device, topic_batch)
        '''
        src_input, src_extend_ids, src_oov, source = [], [], [], []
        for i in indexes:
            src_input.append(self.data[i][0])
            src_extend_ids.append(self.data[i][1])
            src_oov.append(self.data[i][2])
            source.append(self.data[i][3])
        return Batch(self, src_input, src_extend_ids, src_oov, source, device)


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
        #print(lens)
        lens = [len(s) for s in data]
        return data, lens


    def from_txt(self, src_trunc_len=-1):
        """Read and tokenize data from TSV file.
            Args:
                part (str): the name of the part.
            Yields:
                (list[int], list[int]): pairs for each example in dataset.
        """
        with open(self.inputfile+".txt") as file:
            reader = file.readlines()
            for row in reader:
                source = row.strip()
                source_ids = self.sp.EncodeAsIds(row)
                source_words = self.sp.EncodeAsPieces(row)
                source_extend_ids, source_oov = self.src_extend(source_ids[:src_trunc_len], source_words[:src_trunc_len])
                yield tuple([source_ids, source_extend_ids, source_oov, source])

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
                    if token.item() == self.eos_idx :
                        break
                    token_list.append(token.item())
                else:
                    if token == self.eos_idx :
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
                    if token.item() == self.eos_idx :
                        break
                    token_list.append(token.item())
                else:
                    if token == self.eos_idx :
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
    def __init__(self, data_loader, src_input, src_extend_ids, src_oov, source, device, topic=None):
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

        self.__setattr__("src_text", source)

        if topic is not None:
            tensor_topic, topic_length = data_loader.pad(topic, topic_flag=True)
            self.__setattr__("topic", torch.tensor(tensor_topic, dtype=torch.long, device=device))


if __name__ == "__main__":
    bpe_path = config.bpe_model_filename
    dataset = config.dataset
    duc_name = 'DUC2004'
    src_part = 'input'
    loader = EvalBatcher(dataset, duc_name, src_part, bpe_path)
    device = torch.device("cuda")
    batch = loader.next_batch(32, device)
    print(loader.data_length)
    print(loader.max_len)
    print(batch.src)
    duc_name = 'DUC2004'
    directory = os.path.join(dataset, duc_name)
    duc_target = EvalTarget(directory, ['task1_ref0.txt', 'task1_ref1.txt', 'task1_ref2.txt', 'task1_ref3.txt'])
    print(len(duc_target.text))
    print(duc_target.text[0])