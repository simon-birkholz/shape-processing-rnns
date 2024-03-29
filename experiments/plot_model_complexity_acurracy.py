import math

from models.architecture import FeedForwardTower, GammaNetWrapper

import matplotlib as mpl
import matplotlib.pyplot as plt
import collections.abc as abc


def get_model_complexities(cell_types, tower_type):
    result = []
    for cell in cell_types:
        if cell == 'gamma':
            model = GammaNetWrapper(tower_type='small')
            print(f'GammaNet Params: {sum(p.numel() for p in model.parameters())}')
        else:
            model = FeedForwardTower(tower_type=tower_type, cell_type=cell, cell_kernel=3, do_preconv=True,
                                     skip_first=True,
                                     preconv_kernel=1)
        pCount = sum(p.numel() for p in model.parameters())
        result.append(pCount)
    return result


labels = {'conv': 'FF', 'rnn': 'RNN', 'reciprocal': 'RGC', 'gru': 'GRU', 'lstm': 'LSTM', 'hgru': 'hGRU',
          'fgru': 'fGRU', 'gamma': '$\gamma$-Net', 'conv-w' : 'FF (Wider)', 'conv-n': 'FF'}

# ACC = {'conv': 0.4538, 'rnn': {3: 0.5222, 5: 0.5429, 7: 0.5575, 9: 0.5644, 11: 0.5697, 13: 0.572, 15: 0.5726},
#       'reciprocal': {3: 0.5262, 5: 0.5327}}

ACC_FULL = {'conv': 0.4538, 'rnn': {3: 0.5222, 7: 0.5516, 15: 0.5641},
            'reciprocal': {3: 0.5262, 7: 0.552, 15: 0.5213}, 'gru': {3: 0.5285, 7: 0.5867, 15: 0.5604},
            'lstm': {7: 0.5582, 15: 0.3551}, 'hgru': {7: 0.5382, 15: 0.0718}, 'fgru': {7: 0.5196, 15: 0.1248}}

ACC_SINGLE = {'conv': {'n' : 0.4538, 'w' : 0.5243} , 'rnn': 0.5516, 'reciprocal': 0.552, 'gru': 0.5867, 'lstm': 0.5582, 'hgru': 0.5403,
              'fgru': 0.5517, 'gamma': 0.4818}

def get_min_value(data):
    m = math.inf
    for k in data.keys():
        if isinstance(data[k], abc.Mapping):
            m = min(m, get_min_value(data[k]))
        else:
            m = min(m, data[k])
    return m
def drop_keys(data, *dd):
    for d in dd:
        data.pop(d,None)
    return data

def plot_model_complexity_accuracy(data, outfile):

    model_params = {k: get_model_complexities([k], 'normal')[0] for k in data.keys()}

    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })
    plt.figure()

    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]

    factor = 10 ** 6

    for idx, cell_type in enumerate(data.keys()):
        cell_acc = data[cell_type]
        color = colorseq[idx]
        if isinstance(cell_acc, abc.Mapping):
            for ts in cell_acc.keys():


                if cell_type == 'conv':
                    # special chase for conv, because of wider config
                    ts_color = mpl.colors.to_rgba(color, 1.0)
                    if ts == 'w':
                        param_cnt = get_model_complexities(['conv'], 'wider')[0]
                    else:
                        param_cnt = model_params[cell_type]

                    type_label = labels[cell_type + '-' + ts]
                    plt.plot(param_cnt / factor, cell_acc[ts], 'o', label=f'{type_label}',
                             color=ts_color)
                    plt.text(param_cnt / factor, cell_acc[ts], f' {type_label}',
                             horizontalalignment='left',
                             verticalalignment='center', color=ts_color)
                else:
                    ts_color = mpl.colors.to_rgba(color, 1.0 if ts == 7 else 0.4)
                    type_label = labels[cell_type]
                    plt.plot(model_params[cell_type] / factor, cell_acc[ts], 'o', label=f'{type_label} T={ts}', color=ts_color)
                    plt.text(model_params[cell_type] / factor, cell_acc[ts], f' {type_label} T={ts}',
                             horizontalalignment='left',
                             verticalalignment='center', color=ts_color)
        else:
            type_label = labels[cell_type]
            plt.plot(model_params[cell_type] / factor, cell_acc, 'o', label=type_label, color=color)
            plt.text(model_params[cell_type] / factor, cell_acc, f' {type_label}', horizontalalignment='left',
                     verticalalignment='center', color=color)

    plt.xlim(4, 45)
    plt.ylim(get_min_value(data) - 0.05, 0.65)
    plt.xlabel('Number of Parameters x$10^6$')
    plt.ylabel('Accuracy on Imagenet')
    # plt.legend()
    plt.savefig(outfile)


if __name__ == '__main__':
    ext = 'pdf'
    plot_model_complexity_accuracy(ACC_SINGLE, f'../figures/model-complexity-acc.{ext}')

    plot_model_complexity_accuracy(drop_keys(ACC_FULL,'hgru','fgru'), f'../figures/model-complexity-acc-full.{ext}')
