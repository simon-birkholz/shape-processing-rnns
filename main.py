import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import argparse
import json
import os
from pathlib import Path

from typing import Union

from datasets.selection import select_dataset
from models.architecture import FeedForwardTower

from utils import EarlyStopping


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          epochs: Union[int, str],
          aux_cls: bool = True,
          device='cpu'):
    if val_loader:
        print('Detected Validation Dataset')

    aux_discount = 0.4
    do_early_stopping = False
    if epochs == 'early-stop':
        epochs = 100
        do_early_stopping = True
        early_stopping = EarlyStopping(tolerance=5)

    model.to(device)
    for epoch in range(epochs):
        auxiliary_loss = 0.0
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
            if aux_cls:
                outputs, aux_out = model(inputs)
                loss_m = loss_fn(outputs, targets)
                loss_aux = loss_fn(aux_out, targets)
                auxiliary_loss += loss_aux.data.item()
                loss = loss_m + aux_discount * loss_aux
            else:
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
                # Dont care for aux classifier during validation
                if aux_cls:
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.data.item()
                predicted = torch.argmax(outputs, dim=1)
                val_correct += torch.sum(predicted == targets).item()
                val_examples += predicted.shape[0]
            val_loss /= len(val_loader)

            # Logging
            val_accuracy = (val_correct / val_examples)

            log_data = {'training_loss': training_loss, 'train_acc': train_accuracy, 'val_loss': val_loss,
                        'val_acc': val_accuracy, 'auxiliary_loss': auxiliary_loss}
            wandb.log(log_data)
            print(
                f'\nEpoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}, Validation Loss: {val_loss:.2f}, Validation Acc: {val_accuracy:.2f}')
            if do_early_stopping and early_stopping(val_loss):
                print(f'Early stopping after epoch {epoch + 1}')
                return
        else:
            wandb.log({'training_loss': training_loss, 'train_acc': train_accuracy})
            print(f'Epoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}')


def learn(allparams,
          dataset: str,
          dataset_path: str,
          dataset_val_path: str,
          save_dir: str,
          batch_size: int,
          epochs: Union[int, str],
          learning_rate: float,
          model_base: str,
          optimizer: str,
          **config):
    run = wandb.init(
        project='shape-processing-rnns',
        entity='cenrypol',
        config=dict(params=allparams)
    )
    # TODO wandb as context manager
    # TODO dataset selection as context manager (datasets,validation sets and number of classes)
    # TODO checkpoints and saving as context manager

    train_data_loader, val_data_loader, num_classes = select_dataset(dataset, dataset_path, dataset_val_path,
                                                                     batch_size)

    if model_base == 'ff_tower':
        network = FeedForwardTower(num_classes=num_classes, **config)
    elif model_base == 'resnet18':
        network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
    else:
        raise ValueError(f'Unknown base architecture {model_base}')

    if optimizer == 'adam':
        opti = optim.AdamW(network.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        opti = optim.SGD(network.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Unknown optimizer {optimizer}')

    parameter_counter = sum(p.numel() for p in network.parameters())

    print(f'Used network has {parameter_counter} trainable parameters')

    loss = nn.CrossEntropyLoss()

    train(network, opti, loss, train_data_loader, val_data_loader, epochs, 'cuda')

    outpath = f'{save_dir}'
    outparent = Path(save_dir).parent.absolute()

    print(f'Saving model at {outpath}')
    if not os.path.exists(outparent):
        os.makedirs(outparent, exist_ok=True)
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
