import argparse
import json
import os
import random
from typing import Union

import serrelabmodels.gamanet
import torch
import torch.nn.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import wandb
from datasets.selection import select_dataset
from models.architecture import FeedForwardTower, GammaNetWrapper
from utils import EarlyStopping, WBContext, ModelFileContext, get_args_names

from arguments import OPTIMIZERS, LR_SCHEDULER, get_argument_instance
from models.dev_architecture import get_dev_testing_architecture


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          *,
          epochs: Union[int, str],
          aux_cls: bool = False,
          do_gradient_clipping: bool = False,
          batch_frag: int = 1,
          lr_scheduler: str = None,
          lr_step: int = None,
          save_cb=None,
          start_epoch: int = 0,
          loaded_optim: bool = False,
          device: str = 'cpu'):
    if val_loader:
        print('Detected Validation Dataset')

    do_wb = wandb.run is not None
    if not do_wb:
        print('Could not find wandb.run object')
    # else:
    # wandb.watch(model, log='all')

    if do_gradient_clipping:
        clip_value = 0.7

    do_early_stopping = False
    if epochs == 'early-stop':
        epochs = 400
        do_early_stopping = True
        early_stopping = EarlyStopping(tolerance=4)

    if loaded_optim:
        lr_scheduler = get_argument_instance(LR_SCHEDULER, lr_scheduler, optimizer, is_optional=True, step_size=lr_step,
                                             last_epoch=start_epoch)
    else:
        lr_scheduler = get_argument_instance(LR_SCHEDULER, lr_scheduler, optimizer, is_optional=True, step_size=lr_step)
    if lr_scheduler is not None and start_epoch > 0 and not loaded_optim:
        for _ in range(0, start_epoch):
            lr_scheduler.step()
        print(
            f'Stepped to epoch {start_epoch}. Learning rate is now {lr_scheduler.get_last_lr()}')  # When we start at a later epoch we need to adjust the lr scheduler accordingly

    model.to(device)
    for epoch in range(start_epoch, epochs, 1):
        auxiliary_loss = 0.0
        training_loss = 0.0
        val_loss = 0.0
        model.train()
        train_correct = 0
        train_samples = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # backward pass
            loss.backward()

            # weights update
            if ((batch_idx + 1) % batch_frag == 0) or (batch_idx + 1 == len(train_loader)):
                if do_gradient_clipping:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
                optimizer.step()
                optimizer.zero_grad()

            training_loss += loss.data.item()
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
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
                probabilities = F.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                val_correct += torch.sum(predicted == targets).item()
                val_examples += predicted.shape[0]
            val_loss /= len(val_loader)

            # Logging
            val_accuracy = (val_correct / val_examples)

            log_data = {'training_loss': training_loss, 'train_acc': train_accuracy, 'val_loss': val_loss,
                        'val_acc': val_accuracy, 'auxiliary_loss': auxiliary_loss}
            if do_wb:
                wandb.log(log_data)
            print(
                f'\nEpoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}, Validation Loss: {val_loss:.2f}, Validation Acc: {val_accuracy:.2f}')
            if do_early_stopping and early_stopping(val_loss):
                print(f'Early stopping after epoch {epoch + 1}')
                return
        else:
            if do_wb:
                wandb.log({'training_loss': training_loss, 'train_acc': train_accuracy})
            print(f'Epoch {epoch + 1}, Training Loss: {training_loss:.2f}, Training Acc: {train_accuracy:.2f}')
            if do_early_stopping and early_stopping(training_loss):
                print(f'Early stopping (on training loss) after epoch {epoch + 1}')
                return

        if lr_scheduler is not None:
            lr_scheduler.step()

        if save_cb is not None:
            save_cb(epoch + 1)


def learn(dataset: str,
          dataset_path: str,
          dataset_val_path: str,
          save_dir: str,
          batch_size: int,
          learning_rate: float,
          momentum: float,
          model_base: str,
          optimizer: str,
          weight_decay: int = 0,
          batch_frag: int = -1,
          batch_max: int = -1,
          seed: int = 1,
          **config):
    random.seed(seed)

    # calculated hardware requirement
    if batch_max > 0 and batch_max < batch_size:
        batch_frag = batch_size // batch_max

    if batch_frag == -1:
        batch_frag = 1

    intern_batch_size = batch_size // batch_frag
    intern_learning_rate = learning_rate

    # TODO dataset selection as context manager (datasets,validation sets and number of classes)

    train_data_loader, val_data_loader, num_classes = select_dataset(dataset, dataset_path, dataset_val_path,
                                                                     intern_batch_size)
    model_args = get_args_names(FeedForwardTower.__init__)
    gamma_args = get_args_names(GammaNetWrapper.__init__)
    train_args = get_args_names(train)

    model_config = {k: v for k, v in config.items() if k in model_args}
    train_config = {k: v for k, v in config.items() if k in train_args}
    gamma_config = {k: v for k, v in config.items() if k in gamma_args}

    if model_base == 'ff_tower':
        network = FeedForwardTower(num_classes=num_classes, **model_config)
    elif model_base == 'resnet18':
        network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
    elif model_base == 'dev_tower':
        network = get_dev_testing_architecture()
    elif model_base == 'gammanet':
        network = GammaNetWrapper(num_classes=num_classes, **gamma_config)
    else:
        raise ValueError(f'Unknown base architecture {model_base}')

    # summary(network, input_size=(batch_size, 3, 224, 224))

    opti = get_argument_instance(OPTIMIZERS, optimizer, network.parameters(), lr=intern_learning_rate,
                                 weight_decay=weight_decay, momentum=momentum)

    parameter_counter = sum(p.numel() for p in network.parameters())

    print(f'Used network has {parameter_counter} trainable parameters')

    loss = nn.CrossEntropyLoss()

    with ModelFileContext(network, opti, save_dir, device='cuda') as (save_cb, loaded_epoch, loaded_optim):
        train(network, opti, loss, train_data_loader, val_data_loader, batch_frag=batch_frag,
              device='cuda', save_cb=save_cb, start_epoch=loaded_epoch, loaded_optim=loaded_optim, **train_config)


if __name__ == '__main__':
    print(f'CUDA: {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Config file for models', nargs='+')
    parser.add_argument('--out', type=str, help='name for saving the model', default='test_run')
    parser.add_argument('--wbname', type=str, help='if wandb is used this will be the run name', default='none')

    args = parser.parse_args()
    print(f'Executing {len(args.config_file)} configs: {" ".join(args.config_file)}')

    for cfg_file in args.config_file:
        if not os.path.exists(cfg_file):
            raise ValueError(f"Config file {cfg_file} not found")
        with open(cfg_file) as f:
            config = json.load(f)

        if 'dataset_val_path' not in config.keys():
            config['dataset_val_path'] = None

        # Evaluate Hardware Limitations and compensate eventually with accumulated Gradients
        if 'batch_frag' in config.keys() and 'batch_max' in config.keys():
            raise ValueError('batch_frag and batch_max at the same time are not supported')
        elif 'batch_frag' not in config.keys() and 'batch_max' in config.keys():
            config['batch_frag'] = -1
        elif 'batch_frag' in config.keys() and 'batch_max' not in config.keys():
            config['batch_max'] = -1
        elif 'batch_frag' not in config.keys() and 'batch_max' not in config.keys():
            config['batch_frag'] = 1
            config['batch_max'] = -1

        config['save_dir'] = args.out
        config['wb_run_name'] = args.wbname
        if config['wb_run_name'] == 'none':
            config['wb_run_name'] = None

        if 'seed' in config.keys():
            random.seed(int(config['seed']))
        else:
            random.seed(10)

        with WBContext(config) as wb:
            if callable(wb):
                wb(learn)
            else:
                learn(**wb)
