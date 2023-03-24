import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import wandb
import argparse
import json
import os

from datasets.imagenet import get_imagenet, get_imagenet_small, get_imagenet_kaggle
from datasets.ffcv_utils import loader_ffcv_dataset
from datasets.cifar import get_imagenet_cifar10
from models.architecture import FeedForwardTower


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          epochs: int,
          device='cpu'):
    if val_loader:
        print('Detected Validation Dataset')

    model.to(device)
    for epoch in range(epochs):
        training_loss = 0.0
        val_loss = 0.0
        model.train()
        train_correct = 0
        train_samples = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
            predicted = torch.argmax(outputs, dim=1)
            train_correct += torch.sum(predicted == targets).item()
            train_samples += predicted.shape[0]
        training_loss /= len(train_loader)
        train_accuracy = (train_correct / train_samples)

        if val_loader:
            model.eval()
            val_correct = 0
            val_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.data.item()
                predicted = torch.argmax(outputs, dim=1)
                val_correct += torch.sum(predicted == targets).item()
                val_examples += predicted.shape[0]
            val_loss /= len(val_loader)

            # Logging
            val_accuracy = (val_correct / val_examples)

            wandb.log({'training_loss': training_loss, 'train_acc': train_accuracy, 'val_loss': val_loss, 'val_acc': val_accuracy})
            print(
                f'\nEpoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}, Validation Loss: {val_loss:.2f}, Validation Acc: {val_accuracy:.2f}')
        else:
            wandb.log({'training_loss': training_loss, 'train_acc': train_accuracy})
            print(f'Epoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}')


def learn(allparams,
          dataset: str,
          dataset_path: str,
          dataset_val_path: str,
          save_dir: str,
          batch_size: int,
          epochs: int,
          learning_rate: float, **config):
    run = wandb.init(
        project='shape-processing-rnns',
        entity='cenrypol',
        config=dict(params=allparams)
    )

    ds, ds_val = None, None
    if dataset == 'imagenet':
        ds, ds_val = get_imagenet(dataset_path)
    elif dataset == 'imagenet_small':
        ds, ds_val = get_imagenet_small(dataset_path)
    elif dataset == 'imagenet_kaggle':
        ds, ds_val = get_imagenet_kaggle(dataset_path)
    elif dataset == 'ffcv':
        ds = loader_ffcv_dataset(dataset_path, batch_size)
        if dataset_val_path:
            ds_val = loader_ffcv_dataset(dataset_val_path, batch_size)
    elif dataset == 'cifar10':
        ds, ds_val = get_imagenet_cifar10(dataset_path)
    else:
        raise ValueError('Unknown Dataset')

    if dataset == 'ffcv':
        num_classes = 100
    else:
        num_classes = len(ds.classes)

    # train_data_loader, val_data_loader = None, None
    if dataset != 'ffcv':
        train_data_loader = data.DataLoader(ds, batch_size=batch_size)
        val_data_loader = data.DataLoader(ds_val, batch_size=batch_size) if ds_val else None
    else:
        train_data_loader = ds
        val_data_loader = ds_val

    # network = FeedForwardTower(cell_type='conv',num_classes=num_classes)

    network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)

    adam = optim.AdamW(network.parameters(), lr=learning_rate)

    parameter_counter = sum(p.numel() for p in network.parameters())

    print(f'Used network has {parameter_counter} trainable parameters')

    loss = nn.CrossEntropyLoss()

    train(network, adam, loss, train_data_loader, val_data_loader, epochs, 'cuda')

    outpath = f'output/{save_dir}'

    print(f'Saving model at {outpath}')
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    torch.save(network.state_dict(), outpath)

    run.finish()


if __name__ == '__main__':
    print(f'CUDA: {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Config file for models', nargs='+')
    parser.add_argument('--out', type=str, help='name for saving the model', default='test_run')

    args = parser.parse_args()
    print(f'Executing {len(args.config_file)} configs: {" ".join(args.config_file)}')

    for cfg_file in args.config_file:
        if not os.path.exists(cfg_file):
            raise ValueError(f"Config file {cfg_file} not found")
        with open(cfg_file) as f:
            config = json.load(f)

        if 'dataset_val_path' not in config.keys():
            config['dataset_val_path'] = None

        config['save_dir'] = args.out
        allparams = config.copy()
        learn(allparams, **config)
