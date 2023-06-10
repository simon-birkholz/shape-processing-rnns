import math
import wandb
import functools
from typing import Dict
from pathlib import Path
import json
import torch
import os
import inspect

import collections.abc as abc


def traverse_obj(obj, *keys):
    for k in keys:
        if isinstance(obj, abc.Mapping) and k in obj.keys():
            obj = obj[k]
        else:
            return None
    return obj


def get_args_names(fn):
    sign = inspect.getfullargspec(fn)
    return sign.args + sign.kwonlyargs


class EarlyStopping:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.min_criterion = math.inf

    def __call__(self, criterion):
        if self.min_criterion - criterion > 0:
            # criterion decreased, everything alright
            self.counter = 0
            self.min_criterion = criterion
        else:
            self.counter += 1
            if self.counter > self.tolerance:
                return True
        return False


ENTITY = 'cenrypol'
PROJECT = 'shape-processing-rnns'


class WBContext:

    def __init__(self, config: Dict):
        self.entity = ENTITY
        self.project = PROJECT

        self.suppress = config.get('wb_suppress', False)
        self.sweep = config.pop('wb_sweep_id', None)
        config.pop('wb_suppress', None)
        self.params = config.copy()
        self.group = config.pop('wb_group', None)
        self.run_name = config.pop('wb_run_name', None)
        self.config = config

    def __enter__(self):
        if not self.suppress:

            additional_args = dict()
            if self.group:
                additional_args['group'] = self.group

            if self.run_name:
                additional_args['name'] = self.run_name

            if self.sweep is not None:
                print('Detected Sweep Configuration')
                complete_sweep_id = f'{self.entity}/{self.project}/{self.sweep}'

                def train_fun_wrapper(train_fun):
                    self.run = wandb.init(
                        config=dict(params=self.params),
                        **additional_args
                    )
                    updated_values = dict(self.config, **self.run.config)
                    updated_values.pop('params', None)
                    train_fun(**updated_values)

                def agent_wrapper(train_fun):
                    wandb.agent(sweep_id=complete_sweep_id, function=functools.partial(train_fun_wrapper, train_fun),
                                count=1)

                return agent_wrapper
            else:
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    config=dict(params=self.params),
                    **additional_args
                )
                return self.config
        else:
            # jupst pass on config object
            return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.suppress:
            self.run.finish()


def remove_extension(path: Path)-> str:
    return os.path.splitext(path)[0]

class ModelFileContext:

    def __init__(self, network: torch.nn.Module, optim: torch.optim.Optimizer, outpath: str, do_reload=True, interval=5, device='cuda'):
        self.network = network
        self.optim = optim
        self.outpath = Path(outpath)
        self.do_reload = do_reload
        self.checkpoints = []
        self.interval = interval
        self.device = device

    def __enter__(self):
        info_file = Path(f'{remove_extension(self.outpath)}.info')
        prefix = remove_extension(self.outpath)
        loaded = 0
        loaded_optimizer = False
        if self.do_reload and info_file.exists():
            with open(info_file, 'r') as f:
                self.checkpoints = json.load(f)

            best_checkpoint = max(self.checkpoints)
            best_checkpoint_file = f'{prefix}-ep{best_checkpoint}.weights'
            print(f'Loading model from {best_checkpoint_file}')
            state = torch.load(best_checkpoint_file)
            self.network.load_state_dict(state)
            self.network.to(self.device)
            if self.optim is not None:
                try:
                    best_optim_file = f'{prefix}-ep{best_checkpoint}.optim'
                    print(f'Try loading optimizer from {best_optim_file}')
                    optim_state = torch.load(best_optim_file)
                    self.optim.load_state_dict(optim_state)
                    print(f'Loaded optimizer state from {best_optim_file}')
                    loaded_optimizer = True
                except:
                    print(f'Error on loading optimizer state')
            loaded = best_checkpoint
        return self.save_model, loaded, loaded_optimizer

    def save_model(self, epoch: int):
        if epoch % self.interval == 0:
            prefix = remove_extension(self.outpath)
            checkpoint_file = f'{prefix}-ep{epoch}.weights'
            print(f'Saving model at {checkpoint_file}')
            parent = Path(checkpoint_file).parent.absolute()

            if not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            torch.save(self.network.state_dict(), checkpoint_file)
            if self.optim is not None:
                try:
                    optim_file = f'{prefix}-ep{epoch}.optim'
                    torch.save(self.optim.state_dict(), optim_file)
                except:
                    print(f'Error on loading optimizer state')

            self.checkpoints.append(epoch)
            info_file = Path(f'{remove_extension(self.outpath)}.info')
            with open(info_file, 'w') as f:
                json.dump(self.checkpoints, f)
        else:
            # not saving skipping for now
            return

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'Saving model at {self.outpath}')
        parent = Path(self.outpath).parent.absolute()
        if not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        torch.save(self.network.state_dict(), self.outpath)
