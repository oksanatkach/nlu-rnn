import json

from nltk import accuracy

# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_1.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_5.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_10.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_47.json', 'r'))

# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_1.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_5.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_10.json', 'r'))
inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_47.json', 'r'))


# distances = [el['distance'] for el in RNN_BPTT_1]
# print(max(distances))

error_counts = {}
# error_counts = {1: {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0},
#                 5: {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0},
#                 10: {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0},
#                 28: {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}}

for el in inference_results:
    distance = el['distance']

    if distance not in error_counts:
        error_counts[distance] = {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}

    # this_bin = None
    # for key in error_counts.keys():
    #     if distance <= key:
    #         this_bin = key
    #         break

    if el['true_label'] == el['pred_label']:
        if el['true_label'] == 1:
            error_counts[distance]['tp'] += 1
            # error_counts[this_bin]['tp'] += 1
        elif el['true_label'] == 0:
            error_counts[distance]['tn'] += 1
            # error_counts[this_bin]['tn'] += 1

    elif el['true_label'] != el['pred_label']:
        if el['true_label'] == 1:
            error_counts[distance]['fn'] += 1
            # error_counts[this_bin]['fn'] += 1
        elif el['true_label'] == 0:
            error_counts[distance]['fp'] += 1
            # error_counts[this_bin]['fp'] += 1

accuracies = [ (distance, (counts['tp'] + counts['tn']) / sum(counts.values())) for distance, counts in error_counts.items() ]
# accuracies = [ (bin, (counts['tp'] + counts['tn']) / sum(counts.values())) for bin, counts in error_counts.items() ]
accuracies = sorted(accuracies, key=lambda x: x[0])
distances, accuracies = zip(*accuracies)

# labels = ['1', '2-5', '6-10', '11-28']

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
# plt.bar(labels, accuracies)
plt.plot(distances, accuracies)
plt.show()
