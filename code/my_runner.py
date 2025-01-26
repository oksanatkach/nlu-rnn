import numpy as np
import pandas as pd
from utils import invert_dict, load_lm_dataset, docs_to_indices, seqs_to_lmXY
from rnnmath import fraq_loss
from rnn import RNN
from runner import Runner

data_folder = '../data/'
np.random.seed(2018)
train_size = 1000
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

docs = load_lm_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(docs, word_to_num, 1, 1)
X_train, D_train = seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(docs, word_to_num, 1, 1)
X_dev, D_dev = seqs_to_lmXY(S_dev)

X_train = X_train[:train_size]
D_train = D_train[:train_size]
X_dev = X_dev[:dev_size]
D_dev = D_dev[:dev_size]

# q = best unigram frequency from omitted vocab
# this is the best expected loss out of that set
q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

my_runner = Runner(RNN(vocab_size, hdim, vocab_size))
x = X_train[0]
d = D_train[0]
y, s = my_runner.model.predict(x)
# print(my_runner.model.acc_deltas(x, d, y, s))
print(my_runner.model.acc_deltas_bptt(x, d, y, s, 5))
# print(my_runner.model._deltas)
