import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score


def get_losses(text, BPTT):
    text_bptt = re.search(rf'Steps for back propagation: {BPTT}\n(.*?)training finished after reaching maximum of 20 epochs',
                    text,
                    re.DOTALL).groups()[0]
    losses = re.findall(r'epoch.*?new loss: (.*?)	', text_bptt)
    losses = [float(el) for el in losses]
    return losses

def get_times(text, BPTT):
    text_bptt = re.search(rf'Steps for back propagation: {BPTT}\n(.*?)training finished after reaching maximum of 20 epochs',
                    text,
                    re.DOTALL).groups()[0]
    times = re.findall(r'epoch.*?epoch done in (\d*?\.\d*?) ', text_bptt)
    times = [float(el) for el in times]
    return times


def plot_losses(model):
    text = open(f'results/{model}_classifier_BPTT_sweep').read()
    losses_1 = get_losses(text, 1)
    losses_5 = get_losses(text, 5)
    losses_10 = get_losses(text, 10)
    losses_47 = get_losses(text, 47)

    epochs = list(range(1, 21))

    # Plot each model's loss values
    plt.plot(epochs, losses_1, linestyle='-', color='b', label='BPTT 1')
    plt.plot(epochs, losses_5, linestyle='-', color='g', label='BPTT 5')
    plt.plot(epochs, losses_10, linestyle='-', color='r', label='BPTT 10')
    plt.plot(epochs, losses_47, linestyle='-', color='c', label='BPTT 47')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model} Losses Across BPTT Steps')

    # Ensure x-axis shows integer values only
    plt.xticks(range(1, 21))  # Force integer ticks from 1 to 20

    # Add legend to identify the models
    plt.legend()

    # Show plot with grid
    plt.grid(True)
    plt.show()


def plot_times():
    model = 'RNN'
    text = open(f'results/{model}_classifier_BPTT_sweep').read()
    times_1 = get_times(text, 1)
    times_5 = get_times(text, 5)
    times_10 = get_times(text, 10)
    times_47 = get_times(text, 47)

    RNN_total_times = [sum(times_1)/60, sum(times_5)/60, sum(times_10)/60, sum(times_47)/60]

    model = 'GRU'
    text = open(f'results/{model}_classifier_BPTT_sweep').read()
    times_1 = get_times(text, 1)
    times_5 = get_times(text, 5)
    times_10 = get_times(text, 10)
    times_47 = get_times(text, 47)

    GRU_total_times = [sum(times_1) / 60, sum(times_5) / 60, sum(times_10) / 60, sum(times_47) / 60]

    x = np.arange(4)
    width = 0.35
    print(x - width/2)
    plt.bar(x - width/2, RNN_total_times, width, label='Vanilla RNN')
    plt.bar(x + width/2, GRU_total_times, width, label='GRU')
    plt.xlabel('BPTT Steps')
    plt.ylabel('Minutes')
    plt.title('Training Times Across BPTT Steps')
    plt.xticks(x, ['1', '5', '10', '47'])

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_F1s(df):
    BPTTs = [1, 5, 10, 47]
    RNN_F1s = [f1_score(df[f'RNN_{BPTT}_true_label'], df[f'RNN_{BPTT}_pred_label']) for BPTT in BPTTs]
    GRU_F1s = [f1_score(df[f'GRU_{BPTT}_true_label'], df[f'GRU_{BPTT}_pred_label']) for BPTT in BPTTs]

    x = np.arange(4)
    width = 0.35
    print(x - width/2)
    plt.bar(x - width/2, RNN_F1s, width, label='Vanilla RNN')
    plt.bar(x + width/2, GRU_F1s, width, label='GRU')
    plt.xlabel('BPTT Steps')
    plt.ylabel('F1')
    plt.title('F1 Across BPTT Steps')
    plt.xticks(x, ['1', '5', '10', '47'])
    plt.ylim(0.0, 1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heatmap_distance_bins(df):
    df['distance_bin'] = pd.cut(df['distance'], bins=[0, 1, 5, 10, 30], labels=["1", "5", "10", "30"])

    binned_results = []

    for model in ['RNN', 'GRU']:
        for BPTT in [1, 5, 10, 47]:
            for bin in ["1", "5", "10", "30"]:
                binned_results.append(
                    {
                        'model': model,
                        'BPTT': BPTT,
                        'bin': int(bin),
                        'F1': f1_score(df[df['distance_bin'] == bin][f'{model}_{BPTT}_true_label'],
                                       df[df['distance_bin'] == bin][f'{model}_{BPTT}_pred_label'])
                    }
                )

    f1_df = pd.DataFrame(binned_results)

    RNN_f1_df = (f1_df[f1_df['model'] == 'RNN']
                .pivot_table(
                    columns=['bin'],
                    index=['model', 'BPTT'],
                    values=['F1']
                )
                .sort_index(ascending=False))

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(RNN_f1_df,
                annot=True,
                cmap='coolwarm',
                cbar=False
                )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('F1 Scores Across BPTT Steps and Distances')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['1', '5', '10', '30'])
    plt.xlabel('Dependency Distance')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()

    # GRU_f1_df = (f1_df[f1_df['model'] == 'GRU']
    #             .pivot_table(
    #                 columns=['bin'],
    #                 index=['model', 'BPTT'],
    #                 values=['F1']
    #             )
    #             .sort_index(ascending=False))

    # plt.figure(figsize=(8, 7))
    # ax = sns.heatmap(GRU_f1_df,
    #             annot=True,
    #             cmap='coolwarm',
    #             cbar=False
    #             )
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    # plt.title('F1 Scores Across BPTT Steps and Distances')
    # plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['1', '5', '10', '30'])
    # plt.xlabel('Dependency Distance')
    # plt.subplots_adjust(top=0.9, bottom=0.1)
    # plt.show()


def plot_heatmap_seq_length_bins(df):
    df['length'] = df['tokens'].apply(len)
    bins = [1, 5, 10, df['length'].max()]
    df['length_bin'] = pd.cut(df['length'], bins=[0] + bins, labels=bins)

    binned_results = []

    for model in ['RNN', 'GRU']:
        for BPTT in [1, 5, 10, 47]:
            for bin in bins:
                binned_results.append(
                    {
                        'model': model,
                        'BPTT': BPTT,
                        'bin': bin,
                        'F1': f1_score(df[df['length_bin'] == bin][f'{model}_{BPTT}_true_label'],
                                       df[df['length_bin'] == bin][f'{model}_{BPTT}_pred_label'])
                    }
                )
    f1_df = pd.DataFrame(binned_results)

    # RNN_f1_df = (f1_df[f1_df['model'] == 'RNN']
    #             .pivot_table(
    #                 columns=['bin'],
    #                 index=['model', 'BPTT'],
    #                 values=['F1']
    #             )
    #             .sort_index(ascending=False))

    # plt.figure(figsize=(8, 7))
    # ax = sns.heatmap(RNN_f1_df,
    #             annot=True,
    #             cmap='coolwarm',
    #             cbar=False
    #             )
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    # plt.title('F1 Scores Across BPTT Steps and Lengths')
    # plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=bins)
    # plt.xlabel('Sequence Length')
    # plt.subplots_adjust(top=0.9, bottom=0.1)
    # plt.show()

    GRU_f1_df = (f1_df[f1_df['model'] == 'GRU']
                .pivot_table(
                    columns=['bin'],
                    index=['model', 'BPTT'],
                    values=['F1']
                )
                .sort_index(ascending=False))

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(GRU_f1_df,
                annot=True,
                cmap='coolwarm',
                cbar=False
                )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('F1 Scores Across BPTT Steps and Lengths')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['1', '5', '10', '30'])
    plt.xlabel('Sequence Length')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()


def get_df_results():
    df = pd.DataFrame()
    models = ['RNN', 'GRU']
    BPTTs = [1, 5, 10, 47]
    for model in models:
        for BPTT in BPTTs:
            if df.empty:
                df = pd.read_json(f'results/predictions_best_models/Q4_{model}_BPTT_{BPTT}.json')
                df = df.rename(
                    columns={'true_label': f'{model}_{BPTT}_true_label',
                             'pred_label': f'{model}_{BPTT}_pred_label'})
            else:
                temp_df = pd.read_json(f'results/predictions_best_models/Q4_{model}_BPTT_{BPTT}.json')
                temp_df = temp_df.rename(
                    columns={'true_label': f'{model}_{BPTT}_true_label',
                             'pred_label': f'{model}_{BPTT}_pred_label'})
                df = df.join(temp_df[[f'{model}_{BPTT}_true_label', f'{model}_{BPTT}_pred_label']])
    return df


if __name__ == '__main__':
    # model = 'RNN'
    # plot_losses(model)
    # plot_times()

    df = get_df_results()
    # plot_F1s(df)
    plot_heatmap_distance_bins(df)
    # plot_heatmap_seq_length_bins(df)
