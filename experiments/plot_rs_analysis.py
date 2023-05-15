import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from fdr import fdrcorrection
import torch

import argparse

def main(filename: str,
         out_file: str,
         fdr: float,
         show: bool = False,):
    # Load RSA results
    saved_state = torch.load(filename)
    comparisons = saved_state["comparisons"]
    args = saved_state['commandline']

    layer_names = [f'hidden{i + 1}' for i in range(5)]

    comparison_layers = ["image"] + layer_names
    comparison_ds = [ds for ds in args.datasets if ds != 'normal']
    all_ds = comparison_ds + ['normal']

    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })


    # Plot all comparison results in custom bar plot
    xs = np.array(
        [i * (len(all_ds)) + j + 1 for i in range(len(comparison_layers)) for j in range(len(comparison_ds))])
    xticks = [i * len(all_ds) + 0.5 + len(comparison_ds) / 2 for i in range(len(comparison_layers))]
    heights = np.concatenate([comparisons[layer].get_means() for layer in comparison_layers])
    lower_error = heights - np.concatenate(
        [comparisons[layer].get_ci(0.95, test_type="bootstrap")[0] for layer in comparison_layers])
    upper_error = np.concatenate(
        [comparisons[layer].get_ci(0.95, test_type="bootstrap")[1] for layer in comparison_layers]) - heights
    pvals = np.concatenate([c.test_zero(test_type="bootstrap") for c in comparisons.values()])
    significant = fdrcorrection(pvals, Q=fdr)  # control FDR
    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
    colors = [colorseq[i] for _ in range(len(comparisons.keys())) for i in range(len(comparison_ds))]
    legend = [mpl.patches.Patch(color=colorseq[i], label=dset) for i, dset in enumerate(comparison_ds)]
    plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
    plt.scatter(x=xs[significant], y=heights[significant] + 0.1, marker="*", color="black")
    plt.xticks(xticks, labels=comparison_layers)
    plt.ylim(top=1.0, bottom=-0.25)
    plt.legend(handles=legend, loc="upper left")
    plt.ylabel('Methods')
    plt.tight_layout()
    # save figure to file
    plt.savefig(out_file)
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to file with RSA results")
    parser.add_argument("-o", "--out", type=str, required=True, help="Path to file to save figure in.")
    parser.add_argument("--show", action="store_true", help="If true, shows plot of RSA (and pauses script).")
    parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
    args = parser.parse_args()

    main(args.filename, args.out, args.fdr, args.show)