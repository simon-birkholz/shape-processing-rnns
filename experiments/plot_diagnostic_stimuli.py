import argparse
import matplotlib.pyplot as plt

import torch
import numpy as np

from fdr import fdrcorrection


def main(filename: str,
         modelname: str,
         out_file: str,
         fdr: float,
         show: bool = False):
    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False,
        'text.usetex' : True
    })

    # Load RSA results
    saved_state = torch.load(filename)
    accs = saved_state['accuracy']
    args = saved_state['commandline']

    plt.figure()
    acc_list = np.array([accs[ds_name]['acc'] for ds_name in args.datasets])
    pvals = np.array([accs[ds_name]['p-value'] for ds_name in args.datasets])
    significant = fdrcorrection(pvals, Q=fdr)

    rand_distributions = [accs[ds_name]['perm'] for ds_name in args.datasets]
    x_pos = np.arange(len(args.datasets))
    plt.bar(x_pos - 0.2, acc_list, width=0.4)
    plt.violinplot(rand_distributions, positions=x_pos + 0.2, widths=0.4, showmeans=True)
    plt.scatter(x_pos[significant] - 0.2, acc_list[significant] + 0.05, marker="*", c="black")
    plt.xticks(x_pos, args.datasets)
    plt.title(f'Accuracy of {modelname} on diagnostic stimuli')
    plt.savefig('../figures/' + out_file)
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True,
                        help="Path to file with diagnostic stimuli results")
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the model")
    parser.add_argument("-o", "--out", type=str, required=True, help="Path to file to save figure in.")
    parser.add_argument("--show", action="store_true", help="If true, shows plot of accuracies (and pauses script).")
    parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
    args = parser.parse_args()

    main(args.filename, args.name, args.out, args.fdr, args.show)
