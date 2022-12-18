#! /bin/python 

import torch 

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_shape(x):
    # print the shape of a given tensor
    print(x.shape)


# Global variables 
device = get_device()
pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3