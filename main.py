from datasets.imagenet import get_imagenet, get_imagenet_small
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

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device='cpu'):
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

        #model.eval()
        #num_correct = 0
        #num_examples = 0
        #for batch in val_loader:
        #    inputs, targets = batch
        #    inputs = inputs.to(device)
        #    targets = targets.to(device)
        #    outputs = model(inputs)
        #    loss = loss_fn(outputs, targets)
        #    val_loss += loss.data.item()
        #    correct = torch.eq(torch.max(F.softmax(outputs), dim=1)[1], targets).view(-1)
#
        #    num_correct += torch.sum(correct).item()
        #    num_examples += correct.shape[0]
        #val_loss /= len(val_loader)

        #print(f'Epoch {epoch}, Training Loss: {training_loss:.2f}, Validation Loss: {val_loss:.2f}, Accurarcy: {(num_correct/num_examples):.2f}')
        print(f'Epoch {epoch}, Training Loss: {training_loss:.2f}')

def learn(allparams, dataset: str, dataset_path: str, **config):

    run = wandb.init(
        project='shape-processing-rnns',
        config=dict(params=allparams)
    )

    ds = None
    if dataset == 'imagenet':
        ds = get_imagenet(dataset_path)
    elif dataset == 'imagenet_small':
        ds = get_imagenet_small(dataset_path)
    else:
        raise ValueError('Unknown Dataset')

    batch_size = 1024

    train_data_loader = data.DataLoader(ds, batch_size=batch_size)
    # val_data_loader = data.DataLoader(ds_val, batch_size=batch_size)

    simple_network = nn.Sequential(
        nn.Conv2d(3, 20, 5),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(200704, 256)
    )

    adam = optim.AdamW(simple_network.parameters(), lr=0.001)

    loss = nn.CrossEntropyLoss()

    train(simple_network, adam, loss, train_data_loader, None, 20, 'cuda')
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
