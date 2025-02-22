import json

# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_1.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_5.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_10.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_RNN_BPTT_47.json', 'r'))

# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_1.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_5.json', 'r'))
# inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_10.json', 'r'))
inference_results = json.load(open('../results/predictions_best_models/Q4_GRU_BPTT_47.json', 'r'))

results = {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}

for el in inference_results:
    if el['true_label'] == el['pred_label']:
        if el['true_label'] == 1:
            results['tp'] += 1
        elif el['true_label'] == 0:
            results['tn'] += 1

    elif el['true_label'] != el['pred_label']:
        if el['true_label'] == 1:
            results['fn'] += 1
        elif el['true_label'] == 0:
            results['fp'] += 1

print((results['tp'] + results['tn']) / sum(results.values()))
