import argparse
import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb
from datasets.selection import select_dataset
from models.architecture import FeedForwardTower
from utils import EarlyStopping, WBContext


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          *,
          epochs: Union[int, str],
          aux_cls: bool = False,
          batch_frag: int = 1,
          device='cpu'):
    if val_loader:
        print('Detected Validation Dataset')

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
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # backward pass
            loss.backward()

            # weights update
            if ((batch_idx + 1) % batch_frag == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            training_loss += loss.data.item()
            predicted = torch.argmax(outputs, dim=1)
            train_correct += torch.sum(predicted == targets).item()
            train_samples += predicted.shape[0]
        training_loss /= len(train_loader)
        auxiliary_loss /= len(train_loader)
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
          *,
          dataset: str,
          dataset_path: str,
          dataset_val_path: str,
          save_dir: str,
          batch_size: int,
          epochs: Union[int, str],
          learning_rate: float,
          momentum: float,
          model_base: str,
          optimizer: str,
          batch_frag: int =1,
          **config):
    allparams['normalized_lr'] = learning_rate * batch_size  # learning rate is dependent on the batch size
    intern_batch_size = batch_size // batch_frag
    intern_learning_rate = learning_rate
    allparams['internal_batch_size'] = intern_batch_size
    allparams['internal_lr'] = intern_learning_rate

    with WBContext(allparams, config):
        # TODO dataset selection as context manager (datasets,validation sets and number of classes)
        # TODO checkpoints and saving as context manager

        train_data_loader, val_data_loader, num_classes = select_dataset(dataset, dataset_path, dataset_val_path,
                                                                         intern_batch_size)

        if model_base == 'ff_tower':
            network = FeedForwardTower(num_classes=num_classes, **config)
        elif model_base == 'resnet18':
            network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        else:
            raise ValueError(f'Unknown base architecture {model_base}')

        if optimizer == 'adam':
            opti = optim.AdamW(network.parameters(), lr=intern_learning_rate)
        elif optimizer == 'sgd':
            opti = optim.SGD(network.parameters(), lr=intern_learning_rate, momentum=momentum, nesterov=True)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        parameter_counter = sum(p.numel() for p in network.parameters())

        print(f'Used network has {parameter_counter} trainable parameters')

        loss = nn.CrossEntropyLoss()

        train(network, opti, loss, train_data_loader, val_data_loader, epochs=epochs, batch_frag=batch_frag, device='cuda')

        outpath = f'{save_dir}'
        outparent = Path(save_dir).parent.absolute()

        print(f'Saving model at {outpath}')
        if not os.path.exists(outparent):
            os.makedirs(outparent, exist_ok=True)
        torch.save(network.state_dict(), outpath)

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
