import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk
import matplotlib as mpl

labels = {'conv': 'FF', 'rnn': 'RNN', 'reciprocal': 'RGC', 'gru': 'GRU', 'lstm': 'LSTM', 'hgru': 'hGRU',
          'fgru': 'fGRU', 'gamma': '$\gamma$-Net'}

colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]

def man_kendall_test(values):
    return mk.original_test(values).p


def switch_dict_layers(d):
    res = dict()
    for k, v in d.items():
        for k2, vv in v.items():
            res.setdefault(k2, dict())[k] = vv
    return res


def plot_ds_stimuli_accs(ds_name, all_accs, out_file, fdr: float, show=False):
    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })
    plt.figure()

    for idx, (name, accs) in enumerate(all_accs.items()):
        color=colorseq[idx+1]
        p = man_kendall_test(accs[:8])
        l_name = labels[name] + ' (*)' if p < fdr else labels[name]
        xs = np.arange(1, len(accs) + 1, 1)
        plt.plot(xs, accs, label=l_name, color=color)

    plt.legend(loc="upper left")
    plt.ylabel('Top1-Accuracy')
    plt.xlabel('Timesteps')
    plt.title(f'Accuracy over time on {ds_name}')
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

    all_accs['rnn'] = load_val_file('../results/stimuli/2023-07-18-03-28_stimuli_series_rnn-skip-first-ts1-15.pt')
    all_accs['reciprocal'] = load_val_file(
        '../results/stimuli/2023-07-18-06-14_stimuli_series_reciprocal-skip-first-ts1-15.pt')
    all_accs['gru'] = load_val_file('../results/stimuli/2023-07-18-04-14_stimuli_series_gru-skip-first-ts1-15.pt')
    all_accs['lstm'] = load_val_file('../results/stimuli/2023-07-18-05-26_stimuli_series_lstm-skip-first-ts1-15.pt')
    all_accs['hgru'] = load_val_file('../results/stimuli/2023-07-18-07-30_stimuli_series_hgru-skip-first-ts1-15.pt')
    all_accs['fgru'] = load_val_file('../results/stimuli/2023-07-18-08-14_stimuli_series_fgru-skip-first-ts1-15.pt')

    new_accs = switch_dict_layers(all_accs)

    for k, v in new_accs.items():
        vv = {k: [i['acc'] for i in v] for k,v in v.items()}
        plot_ds_stimuli_accs(k, vv, f'../figures/{k}_trend_test.pdf', 0.00833)
