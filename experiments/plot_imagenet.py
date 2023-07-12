import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk

labels = {'conv': 'FF', 'rnn': 'RNN', 'reciprocal': 'RGC', 'gru': 'GRU', 'lstm': 'LSTM', 'hgru': 'hGRU',
          'fgru': 'fGRU', 'gamma': '$\gamma$-Net'}

def man_kendall_test(values):
    return mk.original_test(values).p

def plot_imagenet_accs(all_accs, out_file, fdr: float, show = False):
    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })
    plt.figure()


    for name, accs in all_accs.items():
        p = man_kendall_test(accs[:8])
        l_name = labels[name] + ' (*)' if p < fdr else labels[name]
        xs = np.arange(1, len(accs) +1, 1)
        plt.plot(xs,accs, label=l_name)

    plt.legend(loc="upper left")
    plt.ylabel('Top1-Accuracy')
    plt.xlabel('Timesteps')
    plt.tight_layout()
    plt.title(f'Accuracy over time on imagenet')
    plt.savefig('../figures/' + out_file)
    if show:
        plt.show()

def load_val_file(filename: str):
    # Load RSA results
    saved_state = torch.load(filename)
    acc = saved_state["accuracy"]
    return acc



if __name__ == '__main__':
    all_accs = dict()

    all_accs['rnn'] = load_val_file('../results/imagenetval/2023-06-24-02-43_timeseries_rnn-ts7.pt')
    all_accs['gru'] = load_val_file('../results/imagenetval/2023-07-01-18-53_timeseries_gru-ts7.pt')
    all_accs['lstm'] = load_val_file('../results/imagenetval/2023-07-01-19-58_timeseries_lstm-ts7.pt')
    all_accs['reciprocal'] = load_val_file('../results/imagenetval/2023-07-01-20-42_timeseries_reciprocal-ts7.pt')
    all_accs['hgru'] = load_val_file('../results/imagenetval/2023-07-01-22-30_timeseries_hgru-ts7.pt')
    all_accs['fgru'] = load_val_file('../results/imagenetval/2023-07-01-23-11_timeseries_fgru-ts7.pt')

    plot_imagenet_accs(all_accs,'../figures/imagenet_trend_test.pdf', 0.01)
