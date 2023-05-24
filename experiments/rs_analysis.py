from datasets.pascal_voc import PascalVoc

import torch

import torch.nn.functional as F
import torch.utils.data as data

import rsatoolbox
import matplotlib.pyplot as plt

import numpy as np
import argparse

from typing import List

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower

from ds_transforms import NORMAL, FOREGROUND, SHILOUETTE, FRANKENSTEIN, SERRATED

import datetime

torch.set_grad_enabled(False)


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
        method: str,
        cell_type: str,
        cell_kernel: int,
        time_steps: int,
        weights_file: str,
        batch_size: int,
        normalization: str,
        dropout: float,
        show_rdms=False
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

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=cell_kernel, time_steps=time_steps,
                             normalization=normalization, dropout=dropout)

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    activations = {}
    classes = None
    print("Capturing activations ...")
    for ds, ds_name in all_datasets_full:
        ds_loader = data.DataLoader(ds, batch_size=batch_size)
        print(f"     ... {ds_name} stimuli")
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
            models=models, data=data, method=method, pattern_descriptor='classes'
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
            models=models, data=data, method=method, fitter=fitter, k_rdm=1, boot_type="pattern",
            pattern_descriptor='classes'
        )
    else:
        raise ValueError(f'Unknown method {method}')

    print("     ... calculating comparisons")
    comparisons = {}
    for layer in comparison_layers:
        comparisons[layer] = eval_fun(models[layer], rdms['normal'][layer])

    return {
        "comparisons": comparisons
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Datasets / Diagnostic Stimuli to use as input',
                        nargs='+')
    parser.add_argument('--path', type=str, required=True, help='Path to PascalVOC dataset')
    parser.add_argument('--cell_type', type=str, default='conv', help='Type of (Recurrent) cell to evaluate')
    parser.add_argument('--cell_kernel', type=str, default=3, help='Sizes of cell kernels')
    parser.add_argument('--time_steps', type=int, default=3,
                        help='Amount of timesteps to unroll (for conv defaults to 1)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out', type=str, help='name for the saved data', default='test-run')
    parser.add_argument('--set', type=str, default='trainval', help='PascalVOC image set to use (e.g. train)')
    parser.add_argument('--norm', type=str, default='layernorm', help='Normalization to use')
    parser.add_argument('--drop', type=float, default=0.1,
                        help='Dropout used in the model (not sure if needed for evaluation)')
    parser.add_argument('-b', '--batchsize', type=int, default=16)
    parser.add_argument("--method", type=str, default="fixed",
                        help="How to compare RDMs. Can be 'fixed' (no weighting, use rho-a) or 'weighted' (weighted models, use corr).")
    parser.add_argument("--show_rdms", action="store_true",
                        help="If true, shows plot of RDMs (and pauses script halfway).")
    args = parser.parse_args()

    save_data = main(args.datasets, args.path, args.set, args.method, args.cell_type, args.cell_kernel, args.time_steps,
                     args.weights, args.batchsize, args.norm, args.drop,
                     show_rdms=args.show_rdms)

    save_data['commandline'] = args

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_file = f"../results/rsa/{time}_rsa_{args.out}_{args.method}.pt"
    torch.save(save_data, out_file)
    print(f"Written data to {out_file}")
