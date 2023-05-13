from datasets.pascal_voc import PascalVoc

import torch

import torch.nn.functional as F
import torch.utils.data as data

import matplotlib.pyplot as plt

import numpy as np
import argparse

from typing import List

from datasets.imagenet_classes import get_imagenet_class_mapping
from models.architecture import FeedForwardTower

from ds_transforms import NORMAL, FOREGROUND, SHILOUETTE, FRANKENSTEIN, SERRATED



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
    print(f'Got {val_accuracy:.2f} accuracy on diagnostic stimuli')

    perm_distributions = permutation_test(all_voc_predicitions, all_labels)

    print(f'Got {np.mean(perm_distributions):.2f} mean permutation accuracy on diagnostic stimuli')

    p_val = p_value(perm_distributions, val_accuracy)
    print(f'Got {p_val:.2f} as p-value on diagnostic stimuli')

    return {'acc': val_accuracy, 'perm': perm_distributions, 'perm_acc': np.mean(perm_distributions), 'p-value': p_val}


def main(
        datasets: List[str],
        dataset_path: str,
        set: str,
        cell_type: str,
        weights_file: str,
        out_file: str,
        batch_size: int
):
    normal_ds = PascalVoc(dataset_path, set, transform=NORMAL)
    foreground_ds = PascalVoc(dataset_path, set, transform=FOREGROUND)
    shilouette_ds = PascalVoc(dataset_path, set, transform=SHILOUETTE)
    frankenstein_ds = PascalVoc(dataset_path, set, transform=FRANKENSTEIN)
    serrated_ds = PascalVoc(dataset_path, set, transform=SERRATED)

    all_datasets = [(normal_ds, 'normal'), (foreground_ds, 'foreground'), (shilouette_ds, 'shilouette'),
                    (frankenstein_ds, 'frankenstein'), (serrated_ds, 'serrated')]

    keep_datasets = [(ds, ds_name) for ds, ds_name in all_datasets if ds_name in datasets]

    _, _, imagenet2voc = get_imagenet_class_mapping(dataset_path)

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=3, time_steps=3,
                             normalization='layernorm')

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    accs = {}
    for ds, ds_name in keep_datasets:
        ds_loader = data.DataLoader(ds, batch_size=batch_size)
        print(f'Now processing {ds_name} stimuli')
        accs[ds_name] = classify(model, ds_loader, imagenet2voc, device='cuda')

    plt.figure()
    acc_list = [accs[ds_name]['acc'] for _, ds_name in all_datasets]
    rand_distributions = [accs[ds_name]['perm'] for _, ds_name in all_datasets]
    _, ds_names = zip(*all_datasets)
    x_pos = np.arange(len(all_datasets))
    plt.bar(x_pos - 0.2, acc_list, width=0.4)
    plt.violinplot(rand_distributions, positions=x_pos + 0.2, widths=0.4)

    medians = [np.median(data) for data in rand_distributions]
    plt.scatter(x_pos + 0.2, medians, marker='_', color='tab:blue')
    plt.xticks(x_pos, ds_names)
    plt.savefig(out_file)


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

    args = parser.parse_args()

    main(args.datasets, args.path, args.set, args.cell_type, args.weights, args.out, args.batchsize)
