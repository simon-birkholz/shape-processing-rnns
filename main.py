
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
from models.architecture import FeedForwardTower

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device='cpu'):

    if val_loader:
        print('Detected Validation Dataset')

    model.to(device)
    for epoch in range(epochs):
        training_loss = 0.0
        val_loss = 0.0
        model.train()
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
        training_loss /= len(train_loader)


        if val_loader:
            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.data.item()
                correct = torch.eq(torch.max(outputs, dim=1)[1], targets).view(-1)
    #
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            val_loss /= len(val_loader)

            # Logging
            accuracy = (num_correct/num_examples)

            wandb.log({'training_loss' : training_loss, 'val_loss' : val_loss , 'accuracy': accuracy})
            print(f'Epoch {epoch+1}, Training Loss: {training_loss:.2f}, Validation Loss: {val_loss:.2f}, Accuracy: {accuracy:.2f}')
        else:
            wandb.log({'training_loss': training_loss})
            print(f'Epoch {epoch+1}, Training Loss: {training_loss:.2f}')

def learn(allparams, dataset: str, dataset_path: str, save_dir: str, batch_size: int, **config):

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
    else:
        raise ValueError('Unknown Dataset')

    num_classes = len(ds.classes)

    train_data_loader = data.DataLoader(ds, batch_size=batch_size)

    val_data_loader = data.DataLoader(ds_val, batch_size=batch_size) if ds_val else None

    network = FeedForwardTower(cell_type='conv',num_classes=num_classes)

    adam = optim.AdamW(network.parameters(), lr=0.001)

    parameter_counter = sum(p.numel() for p in network.parameters())

    print(f'Used network has {parameter_counter} trainable parameters')

    loss = nn.CrossEntropyLoss()

    train(network, adam, loss, train_data_loader, val_data_loader, 20, 'cuda')

    outpath = f'output/{save_dir}'

    print(f'Saving model at {outpath}')
    torch.save(network.state_dict(),outpath)

    run.finish()


if __name__ == '__main__':
    print(f'CUDA: {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Config file for models')
    parser.add_argument('--out', type=str, help='name for saving the model', default='test_run')

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise ValueError("No config file found")
    with open(args.config_file) as f:
        config = json.load(f)

    config['save_dir'] = args.out
    allparams = config.copy()
    learn(allparams, **config)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
