from models.architecture import FeedForwardTower

import matplotlib as mpl
import matplotlib.pyplot as plt
import collections.abc as abc


def get_model_complexities(cell_types, tower_type):
    result = []
    for cell in cell_types:

        model = FeedForwardTower(tower_type=tower_type, cell_type=cell, cell_kernel=3, do_preconv=True)
        pCount = sum(p.numel() for p in model.parameters())
        result.append(pCount)
    return result


# ACC = {'conv': 0.4538, 'rnn': {3: 0.5222, 5: 0.5429, 7: 0.5575, 9: 0.5644, 11: 0.5697, 13: 0.572, 15: 0.5726},
#       'reciprocal': {3: 0.5262, 5: 0.5327}}

ACC = {'conv': 0.4538, 'rnn': {3: 0.5222, 7: 0.5575, 11: 0.5697, 15: 0.5726},
       'reciprocal': {3: 0.5262}, 'gru': {3: 0.5285}, 'hgru': {3: 0.5461} , 'fgru' : {3 : 0.5517}}

if __name__ == '__main__':
    model_params = {k: get_model_complexities([k], 'normal')[0] for k in ACC.keys()}

    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })

    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]

    factor = 10 ** 6

    for idx, cell_type in enumerate(ACC.keys()):
        cell_acc = ACC[cell_type]
        color = colorseq[idx]
        if isinstance(cell_acc, abc.Mapping):
            for ts in cell_acc.keys():
                plt.plot(model_params[cell_type] / factor, cell_acc[ts], 'o', label=f'{cell_type} T={ts}', color=color)
                plt.text(model_params[cell_type] / factor, cell_acc[ts], f' {cell_type} T={ts}',
                         horizontalalignment='left',
                         verticalalignment='center')
        else:
            plt.plot(model_params[cell_type] / factor, cell_acc, 'o', label=cell_type, color=color)
            plt.text(model_params[cell_type] / factor, cell_acc, f' {cell_type}', horizontalalignment='left',
                     verticalalignment='center')

    plt.xlim(4, 35)
    plt.ylim(0.45, 0.65)
    plt.xlabel('Number of Parameters x$10^6$')
    plt.ylabel('Accuracy on Imagenet')
    # plt.legend()
    plt.savefig('../figures/model-complexity-acc.png')
