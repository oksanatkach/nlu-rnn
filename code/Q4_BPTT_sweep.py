from notebooks.Q4_hyperparameters import vocab_size, hdim, out_vocab_size, lr, train_size, dev_size, lookbacks
import numpy as np
from gru import GRU
from runner import Runner
import pandas as pd
from utils import invert_dict, load_np_dataset, docs_to_indices, seqs_to_npXY
from rnnmath import fraq_loss

data_folder = 'data'
np.random.seed(2018)

epochs = 20

vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                      names=['count', 'freq'], )
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

# Q4

for lookback in [5, 10, 47]:
    # r = Runner(model=RNN(vocab_size=vocab_size, hidden_dims=hdim, out_vocab_size=out_vocab_size))
    r = Runner(model=GRU(vocab_size=vocab_size, hidden_dims=hdim, out_vocab_size=out_vocab_size))
    r.train_np(
        X=X_train,
        D=D_train,
        X_dev=X_dev,
        D_dev=D_dev,
        epochs=epochs,
        learning_rate=lr,
        back_steps=lookback
    )

    r.model.save_params()

    print('######################################################################')
    print('######################################################################')
