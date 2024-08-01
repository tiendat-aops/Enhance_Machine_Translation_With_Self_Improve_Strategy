from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from typing import List, Literal
import math
from tqdm import tqdm
import sys
import os
import gc
import copy
from rich.console import Console
import warnings
import random
from datetime import datetime
import wandb
from pytz import timezone
from sacremoses import MosesTokenizer, MosesDetokenizer
import re
from pathlib import Path
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from underthesea import text_normalize
import argparse
from configs import Config
import heapq
from sklearn.model_selection import train_test_split

torch.manual_seed(1337)
random.seed(1337)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--src_language", type=str, default="en", help="Source language")
parser.add_argument("--tgt_language", type=str, default="vi", help="Target language")
parser.add_argument("--data_name", type=str, help="Name of dataset")
parser.add_argument("--emb_size", type=int, default=1024, help="Embedding size")
parser.add_argument("--nhead", type=int, default=8, help="Number of heads in multi-head attention")
parser.add_argument("--ffn_hid_dim", type=int, default=2048, help="Feedforward hidden dimension size")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--num_phrases", type=int, default=10, help="Number of training epochs")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
parser.add_argument("--train_size", type=int, default=2976999, help="Training dataset size")
parser.add_argument("--test_size", type=int, default=1270, help="Testing dataset size")
parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search")
parser.add_argument("--warm_up", type=int, default=4000, help="Warm-up steps for learning rate scheduler")
parser.add_argument("--load_checkpoint", action="store_true", help="Load from checkpoint")
parser.add_argument("--use_masked_dataset", action="store_true", help="Use masked dataset")
parser.add_argument("--number_vi_masked", type=int, default=30000, help="Number of Vietnamese sentences to mask")
parser.add_argument("--number_en_masked", type=int, default=20000, help="Number of English sentences to mask")
parser.add_argument("--reverse_translate", action="store_true", help="Use reverse translation")
parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch")
parser.add_argument("--folder_path", type=str, default="/work/dat-nt/Machine_Translation/nmt_custom", help="Folder path")
parser.add_argument("--vocab_path", type=str, default=f"./vocab/PhoMT_EnViT5.pth", help="Vocabulary path")
parser.add_argument("--save_model_path", type=str, default=f"./model/model_IWSLT15_training_phoMT_envit5_v0.pth", help="Model save path")
parser.add_argument("--checkpoint_path", type=str, default=f"./model/model_IWSLT15_training_phoMT_envit5_v0.pth", help="Model save path")
parser.add_argument("--self_improve", action="store_true", help="Training model with self-improve strategy")

# args = parser.parse_args()

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, MASK_IDX = 0, 1, 2, 3, 4
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', '<M>']
# puncs = ['.', '!', '?', ';']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
print('Device:', DEVICE)
cfg = Config(**vars(parser.parse_args()))
print(type(cfg.show_config()))
re_clean_patterns = [
    (re.compile(r"&amp; lt ;.*?&amp; gt ;"), ""),
    (re.compile(r"&amp; lt ;"), "<"),
    (re.compile(r"&amp; gt ;"), ">"),
    (re.compile(r"&amp; amp ; quot ;"), "\""),
    (re.compile(r"&amp; amp ; amp ;"), "&"),
    (re.compile(r"&amp; amp ;"), "&"),
    (re.compile(r"&apos; "), ""),
    (re.compile(r"&apos;"), "'"),
    (re.compile(r"&quot;"), "\""),
    (re.compile(r"&#91;"), ""),
    (re.compile(r"&#93;"), ""),
]

console = Console()
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
