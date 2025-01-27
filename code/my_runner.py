import numpy as np
import pandas as pd
from utils import invert_dict, load_np_dataset, docs_to_indices, seqs_to_npXY
from rnnmath import fraq_loss
from rnn import RNN
from runner import Runner

data_folder = '../data/'
np.random.seed(2018)
train_size = 10000
dev_size = 1000
vocab_size = 2000

hdim = 512
lookback = 1
lr = 0.05

# get the data set vocabulary
vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'])
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

# calculate loss vocabulary words due to vocab_size
fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
print(
    "Retained %d words from %d (%.02f%% of all tokens)\n" % (
        vocab_size, len(vocab), 100 * (1 - fraction_lost)))

# load training data
sents = load_np_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(sents, word_to_num, 0, 0)
X_train, D_train = seqs_to_npXY(S_train)

X_train = X_train[:train_size]
Y_train = D_train[:train_size]

# load development data
sents = load_np_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(sents, word_to_num, 0, 0)
X_dev, D_dev = seqs_to_npXY(S_dev)

X_dev = X_dev[:dev_size]
D_dev = D_dev[:dev_size]

