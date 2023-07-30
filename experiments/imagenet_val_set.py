from datasets.pascal_voc import PascalVoc

import torch

import torch.nn.functional as F
import torch.utils.data as data

import datetime

import numpy as np
import argparse
from tqdm import tqdm
from typing import List

from datasets.ffcv_utils import loader_ffcv_dataset
from models.architecture import FeedForwardTower, GammaNetWrapper


def classify(model,
             data_loader,
             time_steps,
             device: str = 'cpu'):
    model.to(device)

    model.eval()
    val_correct = {t: 0 for t in range(0, time_steps)}
    val_examples = {t: 0 for t in range(0, time_steps)}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, targets = batch
            inputs = inputs.to(device)

            outputs = model(inputs, False, True, time_steps)

            for idx, output_t in enumerate(outputs):
                probabilities = F.softmax(output_t, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                val_correct[idx] += torch.sum(predicted == targets).item()
                val_examples[idx] += predicted.shape[0]

    val_accuracy = {i: (val_correct[i] / val_examples[i]) for i in range(0,time_steps)}

    for i in range(0, time_steps):
        print(
            f"     ... T={i+1} {val_accuracy[i]:.2f} accuracy")

    return val_accuracy


def main(
        data_path: str,
        cell_type: str,
        cell_kernel: int,
        time_steps: int,
        weights_file: str,
        batch_size: int,
        normalization: str,
        dropout: float,
):

    #normal_ds = loader_ffcv_dataset(data_path, batch_size)

    model = FeedForwardTower(tower_type='normal', cell_type=cell_type, cell_kernel=cell_kernel, time_steps=time_steps,
                             normalization=normalization, dropout=dropout, do_preconv=True, skip_first=True,
                             preconv_kernel=1)

    model = GammaNetWrapper(tower_type='small', time_steps=3, num_classes=1000)

    state = torch.load(f'../bw_cluster_weights/{weights_file}')
    model.load_state_dict(state)

    #accs = [classify(model, normal_ds, t, device='cuda') for t in range(1, 16)]
    accs = [v for k,v in classify(model,normal_ds,15,device='cuda').items()]

    print(accs)
    return {
        "accuracy": accs
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--cell_type', type=str, default='conv', help='Type of (Recurrent) cell to evaluate')
    parser.add_argument('--cell_kernel', type=str, default=3, help='Sizes of cell kernels')
    parser.add_argument('--time_steps', type=int, default=3,
                        help='Amount of timesteps to unroll (for conv defaults to 1)')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--out', type=str, help='name for the saved diagram', default='test-run')
    parser.add_argument('--norm', type=str, default='layernorm', help='Normalization to use')
    parser.add_argument('--drop', type=float, default=0.1,
                        help='Dropout used in the model (not sure if needed for evaluation)')
    parser.add_argument('-b', '--batchsize', type=int, default=64)
    args = parser.parse_args()

    save_data = main(args.path, args.cell_type, args.cell_kernel, args.time_steps,
                     args.weights, args.batchsize, args.norm, args.drop)

    save_data['commandline'] = args

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_file = f"../results/imagenetval/{time}_timeseries_{args.out}.pt"
    torch.save(save_data, out_file)
    print(f"Written data to {out_file}")
