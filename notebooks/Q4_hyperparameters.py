train_size = 50000 # ???
dev_size = 1000
vocab_size = 2000
hdim = 50 # like in the article
lr = 0.5
out_vocab_size = 2
lookbacks = [1, 5, 10, 47]

# import numpy as np
# from utils import load_np_dataset

# data_folder = '../data/'
# sents = load_np_dataset(data_folder + '/wiki-train.txt')
# lengths = [len(sent)-1 for sent in sents]
# max_sent_length = max(lengths)
# print(max_sent_length)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.hist(lengths)
# plt.show()
