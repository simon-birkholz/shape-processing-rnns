from datasets.pascal_voc import PascalVoc

import torch

import torch.nn.functional as F
import torch.utils.data as data

import rsatoolbox
import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import argparse

from typing import List

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower

from ds_transforms import NORMAL, FOREGROUND, SHILOUETTE, FRANKENSTEIN, SERRATED
from fdr import fdrcorrection


def permutation_test(probs, labels, n=1000):
    probs = probs.numpy()
    labels = labels.numpy()
    results = np.array([np.mean(np.random.permutation(probs) == labels, dtype=np.float32) for _ in range(n)])
    return results


def p_value(random_dist, model_acc):
    return ((random_dist >= model_acc).sum() + 1) / len(random_dist)


def get_activations(model,
                    data_loader,
                    layer_names: List[str],
                    device: str = 'cpu'):
    model.to(device)
    model.eval()

    all_layers = layer_names + ['image', 'label']

    classes = []
    flatten = torch.nn.Flatten()

    activations = {layer: [] for layer in all_layers}

    for batch in data_loader:
        inputs, targets = batch

        classes.append(targets.numpy())
        activations['image'].append(flatten(inputs).numpy())
        activations['label'].append(F.one_hot(targets, num_classes=21).numpy())

        inputs = inputs.to(device)

        outputs, hidden = model(inputs, return_hidden=True)
        for layer_name, layer_state in zip(layer_names, hidden):
            activations[layer_name].append(flatten(layer_state).cpu().detach().numpy())

    # concatenate all captured activations
    for layer in all_layers:
        activations[layer] = np.concatenate(activations[layer])

    classes = np.concatenate(classes)

    return classes, activations


def main(
        datasets: List[str],
        dataset_path: str,
        set: str,
        cell_type: str,
        weights_file: str,
        out_file: str,
        batch_size: int,
        fdr: int,
        method: str,
        show_rdms: bool = False,
        show_rsa: bool = False
):
    normal_ds = PascalVoc(dataset_path, set, transform=NORMAL)
    foreground_ds = PascalVoc(dataset_path, set, transform=FOREGROUND)
    shilouette_ds = PascalVoc(dataset_path, set, transform=SHILOUETTE)
    frankenstein_ds = PascalVoc(dataset_path, set, transform=FRANKENSTEIN)
    serrated_ds = PascalVoc(dataset_path, set, transform=SERRATED)

    all_datasets_full = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                         (frankenstein_ds, 'frankenstein'), (serrated_ds, 'serrated')]

    layer_names = [f'hidden{i + 1}' for i in range(5)]
    all_layers = layer_names + ['image', 'label']
    comparison_layers = ["image"] + layer_names

    all_datasets_full = [(ds, ds_name) for ds, ds_name in all_datasets_full if ds_name in datasets]
    all_datasets = [ds_name for _, ds_name in all_datasets_full]
    comparison_ds = [ds for ds in all_datasets if ds != 'normal']

    _, _, imagenet2voc = get_imagenet_class_mapping(dataset_path)

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=3, time_steps=3,
                             normalization='layernorm')

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    activations = {}
    classes = None
    for ds, ds_name in all_datasets_full:
        ds_loader = data.DataLoader(ds, batch_size=batch_size)
        print(f'Now capturing {ds_name} activations')
        classes, ac = get_activations(model, ds_loader, layer_names, device='cuda')
        activations[ds_name] = ac

    # taken from https://github.com/cJarvers/shapebias/blob/main/experiments/rsa_analysis.py

    print("Performing RSA ...")
    print("     ... creating datasets")

    rsa_datasets = {}
    for ds_name in all_datasets:
        rsa_datasets[ds_name] = {}
        for layer in all_layers:
            activation = activations[ds_name][layer]
            rsa_datasets[ds_name][layer] = rsatoolbox.data.Dataset(
                activation,
                obs_descriptors={'classes': classes}
            )
            rsa_datasets[ds_name][layer].sort_by('classes')
            del activations[ds_name][layer]
        del activations[ds_name]
    del activations

    print("     ... calculating RDMs")

    rdms = {ds_name: {layer: rsatoolbox.rdm.calc_rdm(rsa_datasets[ds_name][layer], method='euclidean')
                      for layer in all_layers}
            for ds_name in all_datasets}

    if show_rdms:
        print("     ... plotting RDMs")
        rows = len(all_datasets)
        cols = len(all_layers)

        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=rows, ncols=1)
        for r, ds_name in enumerate(all_datasets):
            subfig = subfigs[r]
            subfig.suptitle(ds_name)
            axs = subfig.subplots(nrows=1, ncols=cols)
            for c, layer in enumerate(all_layers):
                rsatoolbox.vis.rdm_plot.show_rdm_panel(rdms[ds_name][layer], ax=axs[c], rdm_descriptor=layer)
        plt.savefig('temp-rdms.pdf')
        plt.show()

    if method == 'fixed':
        models = {layer: [rsatoolbox.model.ModelFixed(ds_name, rdms[ds_name][layer]) for ds_name in comparison_ds] for
                  layer in comparison_layers}
        method = 'rho-a'
        method_string = r'rank-correlation $(\rho_a)$'
        eval_fun = lambda models, data: rsatoolbox.inference.eval_bootstrap_pattern(
            models=models, data=data, method=method
        )
    elif method == 'weighted':
        models = {
            layer: [
                rsatoolbox.model.ModelWeighted(ds_name, rdms[ds_name][layer]) for ds_name in comparison_ds
            ] for layer in comparison_layers
        }
        fitter = rsatoolbox.model.fitter.fit_optimize_positive
        method = 'corr'
        method_string = r'linear correlation $(r)$'
        eval_fun = lambda models, data: rsatoolbox.inference.bootstrap_crossval(
            models=models, data=data, method=method, fitter=fitter, k_rdm=1, boot_type="pattern"
        )
    else:
        raise ValueError(f'Unknown method {method}')

    print("     ... calculating comparisons")
    comparisons = {}
    for layer in comparison_layers:
        comparisons[layer] = eval_fun(models[layer], rdms['normal'][layer])

    # Plot all comparison results in custom bar plot
    xs = np.array(
        [i * (len(all_datasets)) + j + 1 for i in range(len(comparison_layers)) for j in range(len(args.comparisons))])
    xticks = [i * len(all_datasets) + 0.5 + len(args.comparisons) / 2 for i in range(len(comparison_layers))]
    heights = np.concatenate([comparisons[layer].get_means() for layer in comparison_layers])
    lower_error = heights - np.concatenate(
        [comparisons[layer].get_ci(0.95, test_type="bootstrap")[0] for layer in comparison_layers])
    upper_error = np.concatenate(
        [comparisons[layer].get_ci(0.95, test_type="bootstrap")[1] for layer in comparison_layers]) - heights
    pvals = np.concatenate([c.test_zero(test_type="bootstrap") for c in comparisons.values()])
    significant = fdrcorrection(pvals, Q=fdr)  # control FDR
    colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
    colors = [colorseq[i] for _ in range(len(comparisons.keys())) for i in range(len(args.comparisons))]
    legend = [mpl.patches.Patch(color=colorseq[i], label=dset) for i, dset in enumerate(args.comparisons)]
    plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
    plt.scatter(x=xs[significant], y=heights[significant] + 0.1, marker="*", color="black")
    plt.xticks(xticks, labels=comparison_layers)
    plt.ylim(top=1.0, bottom=-0.25)
    plt.legend(handles=legend, loc="upper left")
    plt.ylabel(method_string)
    plt.tight_layout()
    # save figure to file
    plt.savefig(out_file)
    if show_rsa:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Datasets / Diagnostic Stimuli to use as input',
                        nargs='+')
    parser.add_argument('--path', type=str, required=True, help='Path to PascalVOC dataset')
    parser.add_argument('--cell_type', type=str, default='conv', help='Type of (Recurrent) cell to evaluate')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out', type=str, help='name for the saved diagram', default='output-diagram.png')
    parser.add_argument('--set', type=str, default='trainval', help='PascalVOC image set to use (e.g. train)')
    parser.add_argument('-b', '--batchsize', type=int, default=16)
    parser.add_argument("--method", type=str, default="fixed",
                        help="How to compare RDMs. Can be 'fixed' (no weighting, use rho-a) or 'weighted' (weighted models, use corr).")
    parser.add_argument("--show_rdms", action="store_true",
                        help="If true, shows plot of RDMs (and pauses script halfway).")
    parser.add_argument("--show_rsa", action="store_true", help="If true, shows plot of RSA (and pauses script).")

    parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
    args = parser.parse_args()

    main(args.datasets, args.path, args.set, args.cell_type, args.weights, args.out, args.batchsize, args.fdr,
         args.method,
         show_rdms=args.show_rdms, show_rsa=args.show_rsa)
