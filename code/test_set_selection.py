from utils import invert_dict
from rnn import RNN
from gru import GRU
from notebooks.Q4_hyperparameters import hdim, out_vocab_size, vocab_size
import numpy as np
import pandas as pd
import json


def sent_with_dependency_lengths():
    vocab = pd.read_table("../data/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                          names=['count', 'freq'], )
    num_to_word = dict(enumerate(vocab.index[:vocab_size]))
    word_to_num = invert_dict(num_to_word)

    sents = []
    with open('../data/wiki-test.txt') as f:
        next(f)
        for line in f:
            text, noun_idx, verb_idx, verb_pos, _, _ = tuple(line.strip().split('\t'))
            noun_idx = int(noun_idx)
            verb_idx = int(verb_idx)
            tokens = text.split()[:verb_idx]
            tokens = [w if w in word_to_num else 'UNK' for w in tokens]
            token_ids = [word_to_num[w] for w in tokens]
            sents.append({'tokens': tokens, 'token_ids': token_ids, 'distance': verb_idx-noun_idx, 'true_label': word_to_num[verb_pos]})

    return sents

test_set = sent_with_dependency_lengths()

# model = RNN(vocab_size=vocab_size, hidden_dims=hdim, out_vocab_size=out_vocab_size)
model = GRU(vocab_size=vocab_size, hidden_dims=hdim, out_vocab_size=out_vocab_size)
BPTT_steps = 47
best_epoch = 16
# best_model_path = f'../results/Q4_RNN/BPTT_{BPTT_steps}/epoch_{best_epoch-1}'
best_model_path = f'../results/Q4_GRU/BPTT_{BPTT_steps}/epoch_{best_epoch-1}'
model.Vr = np.load(f'{best_model_path}/Vr.npy')
model.Vh = np.load(f'{best_model_path}/Vh.npy')
model.Vz = np.load(f'{best_model_path}/Vz.npy')

model.Ur = np.load(f'{best_model_path}/Ur.npy')
model.Uh = np.load(f'{best_model_path}/Uh.npy')
model.Uz = np.load(f'{best_model_path}/Uz.npy')

model.W = np.load(f'{best_model_path}/W.npy')

for el in test_set:
    pred_dists, _ = model.predict(np.array(el['token_ids']))
    el['pred_label'] = int(np.argmax(pred_dists[-1]))

# json.dump(test_set, open(f'../results/predictions_best_models/Q4_RNN_BPTT_{BPTT_steps}.json', 'w'), indent=4)
json.dump(test_set, open(f'../results/predictions_best_models/Q4_GRU_BPTT_{BPTT_steps}.json', 'w'), indent=4)
