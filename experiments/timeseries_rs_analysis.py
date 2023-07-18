import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List

import numpy as np
from fdr import fdrcorrection
from pathlib import Path
import torch

import argparse
import pymannkendall as mk
from tqdm import tqdm


def man_kendall_test(comp_list, comparison_layers):
    values = []
    for comp_t in comp_list:
        values.append(np.concatenate([comp_t[layer].get_means() for layer in comparison_layers]))
    values = np.stack(values)
    values = np.transpose(values)

    res = [mk.original_test(v).p for v in values]
    return res


def man_kendall_test_multiple(comp_list, comparison_layers, K=1000):
    base_p = man_kendall_test(comp_list, comparison_layers)

    res = []
    for k in tqdm(range(K)):
        values = []
        for comp_t in comp_list:
            values.append(np.concatenate([comp_t[layer].evaluations[k] for layer in comparison_layers]))
        values = np.stack(values)
        values = np.transpose(values)

        res.append(np.array([mk.original_test(v).p for v in values]))

    res = np.stack(res)

    # calculating p value
    p_values = ((res >= base_p).sum(axis=0) + 1) / K

    return p_values


def plot_timeseries_rsa(args,
                        modelname: str,
                        comparison_layer,
                        comparison_ds,
                        timesteps,
                        comparisons,
                        out_file: str,
                        fdr: float,
                        show: bool = False):
    plt.rcParams.update({
        'font.size': 15,
        'figure.figsize': (8, 5),
        'axes.spines.right': False,
        'axes.spines.top': False
    })
    plt.figure()

    if args.method == "fixed":
        method_string = r'rank-correlation $(\rho_a)$'
    elif args.method == "weighted":
        method_string = r'linear correlation $(r)$'

    # Plot all comparison results in custom bar plot
    xs = np.array(
        [i * (timesteps + 1) + j + 1 for i in range(len(comparison_ds)) for j in range(timesteps)])
    xticks = [i * (timesteps + 1) + 1 + 0.5 + timesteps / 2 for i in range(len(comparison_ds))]
    heights = np.array(
        [comparisons[t][comparison_layer].get_means()[ds + 2] for ds in range(len(comparison_ds)) for t in
         range(timesteps)])

    lower_error = heights - np.array(
        [comparisons[t][comparison_layer].get_ci(0.95, test_type="bootstrap")[0][ds + 2] for ds in range(len(comparison_ds)) for t in
         range(timesteps)])
    upper_error = np.array(
        [comparisons[t][comparison_layer].get_ci(0.95, test_type="bootstrap")[1][ds + 2] for ds in range(len(comparison_ds)) for t in
         range(timesteps)]) - heights

    pvals = man_kendall_test(comparisons, [comparison_layer])[2:]
    print(pvals)
    significant = fdrcorrection(pvals, Q=fdr)  # control FDR
    # needed_ps = [all_layers.index(l) * (len(comparison_ds)) + j for i, l in enumerate(comparison_layers) for j in
    #             range(len(comparison_ds))]
    # significant = [significant[i] for i in needed_ps]
    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
    colors = [colorseq[i + 2] for i in range(len(comparison_ds)) for _ in range(timesteps)]
    legend = [mpl.patches.Patch(color=colorseq[i + 2], label=dset) for i, dset in enumerate(comparison_ds)]
    plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
    # plt.scatter(x=xs[significant], y= 0.6, marker="*", color="red")
    plt.xticks(xticks, labels=comparison_ds)
    plt.ylim(top=1.0, bottom=-0.25)
    plt.legend(handles=legend, loc="upper left")
    plt.ylabel(method_string)
    plt.tight_layout()
    plt.title(f'Similarity to image representations in {modelname}')
    # save figure to file
    plt.savefig('../figures/' + out_file)
    if show:
        plt.show()


def main(filenames: str,
         modelname: str,
         out_file: str,
         fdr: float,
         show: bool = False):
    # Load RSA results
    comparisons = []
    args = None
    for file in filenames:
        saved_state = torch.load(file)
        comparisons.append(saved_state["comparisons"])
        args = saved_state['commandline']

    layer_names = [f'hidden{i + 1}' for i in range(5)]
    cell_names = [f'cell{i + 1}' for i in range(5)]

    comparison_ds = ['frankenstein', 'serrated']

    plot_timeseries_rsa(args, modelname, 'hidden4', comparison_ds, args.time_steps, comparisons, out_file, fdr, show)
    if args.cell_type in ['reciprocal', 'lstm']:
        out_file2 = Path(out_file).stem + '_cell' + Path(out_file).suffix
        plot_timeseries_rsa(args, modelname, 'cell4', comparison_ds, args.time_steps, comparisons, out_file2, fdr,
                            show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to file with RSA results",
                        action='append')
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the model")
    parser.add_argument("-o", "--out", type=str, required=True, help="Path to file to save figure in.")
    parser.add_argument("--show", action="store_true", help="If true, shows plot of RSA (and pauses script).")
    parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
    args = parser.parse_args()

    main(args.filename, args.name, args.out, args.fdr, args.show)
