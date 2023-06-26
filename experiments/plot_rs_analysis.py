import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List

import numpy as np
from fdr import fdrcorrection
from pathlib import Path
import torch

import argparse


def plot_rsa(args,
             modelname: str,
             comparison_layers: List[str],
             comparison_ds,
             all_ds,
             comparisons,
             out_file: str,
             fdr: float,
             show: bool = False):
    plt.figure()
    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })

    if args.method == "fixed":
        method_string = r'rank-correlation $(\rho_a)$'
    elif args.method == "weighted":
        method_string = r'linear correlation $(r)$'

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
    all_layers = [c for c in comparisons.keys()]
    significant = fdrcorrection(pvals, Q=fdr)  # control FDR
    needed_ps = [all_layers.index(l) * (len(comparison_ds)) + j for i, l in enumerate(comparison_layers) for j in range(len(comparison_ds))]
    significant = [significant[i] for i in needed_ps]
    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
    colors = [colorseq[i] for _ in range(len(comparisons.keys())) for i in range(len(comparison_ds))]
    legend = [mpl.patches.Patch(color=colorseq[i], label=dset) for i, dset in enumerate(comparison_ds)]
    plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
    plt.scatter(x=xs[significant], y=heights[significant] + 0.1, marker="*", color="black")
    plt.xticks(xticks, labels=comparison_layers)
    plt.ylim(top=1.0, bottom=-0.25)
    plt.legend(handles=legend, loc="upper left")
    plt.ylabel(method_string)
    plt.tight_layout()
    plt.title(f'Similarity to image representations in {modelname}')
    # save figure to file
    plt.savefig('../figures/' + out_file)
    if show:
        plt.show()


def main(filename: str,
         modelname: str,
         out_file: str,
         fdr: float,
         show: bool = False):
    # Load RSA results
    saved_state = torch.load(filename)
    comparisons = saved_state["comparisons"]
    args = saved_state['commandline']

    layer_names = [f'hidden{i + 1}' for i in range(5)]
    cell_names = [f'cell{i + 1}' for i in range(5)]

    comparison_layers = ["image"] + layer_names
    if args.cell_type in ['reciprocal', 'lstm']:
        comparison_layers2 = ["image"] + cell_names
    comparison_ds = [ds for ds in args.datasets if ds != 'normal']
    all_ds = comparison_ds + ['normal']

    plot_rsa(args, modelname, comparison_layers, comparison_ds, all_ds, comparisons, out_file, fdr, show)
    if args.cell_type in ['reciprocal', 'lstm']:
        out_file2 = Path(out_file).stem + '_cell' + Path(out_file).suffix
        plot_rsa(args, modelname, comparison_layers2, comparison_ds, all_ds, comparisons, out_file2, fdr, show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to file with RSA results")
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the model")
    parser.add_argument("-o", "--out", type=str, required=True, help="Path to file to save figure in.")
    parser.add_argument("--show", action="store_true", help="If true, shows plot of RSA (and pauses script).")
    parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
    args = parser.parse_args()

    main(args.filename, args.name, args.out, args.fdr, args.show)
