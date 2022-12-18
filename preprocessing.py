#! /bin/python 

import argparse
import re 
import pickle 
import random 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
from utils import *


class Vocabulary:
    def __init__(self) -> None:
        self.word_to_id = {pad_word: pad_id, bos_word: bos_id, eos_word:eos_id, unk_word: unk_id}
        self.word_count = {}
        self.id_to_word = {pad_id: pad_word, bos_id: bos_word, eos_id: eos_word, unk_id: unk_word}
        self.num_words = 4

    def find_tokens(self, sentence):
        # print(sentence)
        res = re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", sentence.strip().lower())
        return res 
    
    def get_ids_from_sentence(self, sentence):
        tokens = self.find_tokens(sentence)
        sent_ids = [bos_id] + [self.word_to_id[word] if word in self.word_to_id \
                               else unk_id for word in tokens] + \
                               [eos_id]
        return sent_ids

    def decode_sentence_from_ids(self, sent_ids):
        words = list()
        for i, word_id in enumerate(sent_ids):
            if word_id in [bos_id, eos_id, pad_id]:
                # Skip these words
                continue
            else:
                words.append(self.id_to_word[word_id])
        return ''.join(words)


    def add_words_from_sentence(self, sentence):
        # print(sentence)
        tokens = self.find_tokens(sentence)
        # print(tokens)
        for word in tokens:
            if word not in self.word_to_id:
                # add this word to the vocabulary
                self.word_to_id[word] = self.num_words
                self.id_to_word[self.num_words] = word
                self.word_count[word] = 1
                self.num_words += 1
            else:
                # update the word count
                self.word_count[word] += 1


class PolynomialDataset(Dataset):
    def __init__(self, all_examples, vocab) -> None:
        super().__init__()
        self.all_examples = all_examples
        self.vocab = vocab 

        def encode(src, tgt):
            src_ids = self.vocab.get_ids_from_sentence(src)
            tgt_ids = self.vocab.get_ids_from_sentence(tgt)
            return (src_ids, tgt_ids)

        # We will pre-tokenize the examples and save in id lists for later use
        self.tokenized_examples = [encode(src, tgt) for src, tgt in self.all_examples]

    def __len__(self):
        return len(self.all_examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"example_ids":self.tokenized_examples[idx], "example":self.all_examples[idx]}


def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    """
    # Sort conv_ids based on decreasing order of the src_lengths.
    # This is required for efficient GPU computations.
    src_ids = [torch.LongTensor(e["example_ids"][0]) for e in data]
    tgt_ids = [torch.LongTensor(e["example_ids"][1]) for e in data]
    src_str = [e["example"][0] for e in data]
    tgt_str = [e["example"][1] for e in data]
    data = list(zip(src_ids, tgt_ids, src_str, tgt_str))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_ids, tgt_ids, src_str, tgt_str = zip(*data)

    # Pad the src_ids and tgt_ids using token pad_id to create src_seqs and tgt_seqs
    src_seqs = nn.utils.rnn.pad_sequence(src_ids, padding_value=pad_id)
    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_ids, padding_value=pad_id)

    return {"example_ids":(src_ids, tgt_ids), "example":(src_str, tgt_str), "example_tensors":(src_seqs.to(device), tgt_seqs.to(device))}


def test_train_split(data, train_ratio):
    random.shuffle(data)
    train_split = int(train_ratio * len(data))
    val_split = int((train_ratio/2 + 0.5)*len(data))
    return data[:train_split], data[train_split:val_split], data[val_split:]


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, 
                               shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create train test split.")
    parser.add_argument("--datapath", type=str, default="dataset.txt")
    # Val and test would be equal splits of the remaining data after removing train. 
    parser.add_argument("--train_perc", type=float, default=0.98)
    args = parser.parse_args()

    with open(args.datapath, 'r') as f:
        all_examples = f.readlines()

    train_ex, val_ex, test_ex = test_train_split(all_examples, args.train_perc)

    # Write the data to file. 
    with open('train.txt', 'w') as f:
        f.write(''.join(train_ex))

    with open('val.txt', 'w') as f:
        f.write(''.join(val_ex))

    with open('test.txt', 'w') as f:
        f.write(''.join(test_ex))

    print("Created the following splits: ")
    print("train: {} lines\nval: {} lines\ntest: {} lines".format(len(train_ex), len(val_ex), len(test_ex)))

    # Create te vocabulary on the complete dataset. 
    vocab = Vocabulary()

    for line in all_examples:
        src, target = line.strip().split('=')
        # print(src, target)
        vocab.add_words_from_sentence(src)
        vocab.add_words_from_sentence(target)
    print(f"Total words in the vocabulary = {vocab.num_words}")
    
    # Dump vocabulary file for later use 
    with open('vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)

    print("Dumped vocabulary to vocab.pkl for later use.")