from datasets.pascal_voc import PascalVoc

import torch

import torch.nn.functional as F
import torch.utils.data as data

import datetime

import numpy as np
import argparse

from typing import List

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower

from ds_transforms import NORMAL, FOREGROUND, SHILOUETTE, FRANKENSTEIN, SERRATED, DEV_TEST


def permutation_test(probs, labels, n=1000):
    probs = probs.numpy()
    labels = labels.numpy()
    results = np.array([np.mean(np.random.permutation(probs) == labels, dtype=np.float32) for _ in range(n)])
    return results


def p_value(random_dist, model_acc):
    return ((random_dist >= model_acc).sum() + 1) / len(random_dist)


def classify(model,
             data_loader,
             imagenet2voc,
             device: str = 'cpu'):
    model.to(device)

    model.eval()
    val_correct = 0
    val_examples = 0
    all_voc_predicitions = torch.empty((0), dtype=torch.int32)
    all_labels = torch.empty((0), dtype=torch.int32)
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            all_labels = torch.cat((all_labels, targets), dim=0)

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            voc_predicted = torch.as_tensor([imagenet2voc[p] for p in predicted], dtype=torch.int32)
            voc_predicted = voc_predicted.to('cpu')

            all_voc_predicitions = torch.cat((all_voc_predicitions, voc_predicted), dim=0)

            val_correct += torch.sum(voc_predicted == targets).item()
            val_examples += predicted.shape[0]

    val_accuracy = (val_correct / val_examples)
    perm_distributions = permutation_test(all_voc_predicitions, all_labels)

    p_val = p_value(perm_distributions, val_accuracy)

    print(
        f"     ... {val_accuracy:.2f} accuracy, {np.mean(perm_distributions):.2f} mean permutation accuracy, {p_val:.2f} as p-value")

    return {'acc': val_accuracy, 'perm': perm_distributions, 'perm_acc': np.mean(perm_distributions), 'p-value': p_val}


def main(
        datasets: List[str],
        dataset_path: str,
        set: str,
        cell_type: str,
        cell_kernel: int,
        time_steps: int,
        weights_file: str,
        batch_size: int,
        normalization: str,
        dropout: float,
):
    normal_ds = PascalVoc(dataset_path, set, transform=NORMAL)
    foreground_ds = PascalVoc(dataset_path, set, transform=FOREGROUND)
    shilouette_ds = PascalVoc(dataset_path, set, transform=SHILOUETTE)
    frankenstein_ds = PascalVoc(dataset_path, set, transform=FRANKENSTEIN)
    serrated_ds = PascalVoc(dataset_path, set, transform=SERRATED)

    all_datasets = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                    (frankenstein_ds, 'frankenstein'), (serrated_ds, 'serrated')]

    all_datasets = [(ds, ds_name) for ds, ds_name in all_datasets if ds_name in datasets]

    _, _, imagenet2voc = get_imagenet_class_mapping(dataset_path)

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=cell_kernel, time_steps=time_steps,
                             normalization=normalization, dropout=dropout, do_preconv=True, skip_first=True, preconv_kernel=1)

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    accs = {}
    for ds, ds_name in all_datasets:
        ds_loader = data.DataLoader(ds, batch_size=batch_size)
        print(f'Processing {ds_name} stimuli...')
        accs[ds_name] = classify(model, ds_loader, imagenet2voc, device='cuda')

    return {
        "accuracy": accs
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
    parser.add_argument('--out', type=str, help='name for the saved diagram', default='test-run')
    parser.add_argument('--set', type=str, default='trainval', help='PascalVOC image set to use (e.g. train)')
    parser.add_argument('--norm', type=str, default='layernorm', help='Normalization to use')
    parser.add_argument('--drop', type=float, default=0.1, help='Dropout used in the model (not sure if needed for evaluation)')
    parser.add_argument('-b', '--batchsize', type=int, default=64)
    args = parser.parse_args()

    save_data = main(args.datasets, args.path, args.set, args.cell_type, args.cell_kernel, args.time_steps,
                     args.weights, args.batchsize, args.norm, args.drop)

    save_data['commandline'] = args

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_file = f"../results/stimuli/{time}_stimuli_{args.out}.pt"
    torch.save(save_data, out_file)
    print(f"Written data to {out_file}")
