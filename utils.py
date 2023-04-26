import math
import wandb
import functools
from typing import Dict
from pathlib import Path
import json
import torch
import os
import inspect
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

        self.suppress = config.get('wb_suppress') is not None
        self.sweep = config.pop('wb_sweep_id', None)
        config.pop('wb_suppress', None)
        self.params = config.copy()
        self.group = config.pop('wb_group', None)
        self.config = config

    def __enter__(self):
        if not self.suppress:
            if self.sweep is not None:
                print('Detected Sweep Configuration')
                complete_sweep_id = f'{self.entity}/{self.project}/{self.sweep}'
                def train_fun_wrapper(train_fun):
                    self.run = wandb.init(
                        group=self.group,
                        config=dict(params=self.params)
                    )
                    updated_values = dict(self.config,**self.run.config)
                    updated_values.pop('params',None)
                    train_fun(**updated_values)
                def agent_wrapper(train_fun):
                    wandb.agent(sweep_id=complete_sweep_id, function=functools.partial(train_fun_wrapper,train_fun), count=1)
                return agent_wrapper
            else:
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    group=self.group,
                    config=dict(params=self.params)
                )
                return self.config
        else:
            # jupst pass on config object
            return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.suppress:
            self.run.finish()

class ModelFileContext:
    def __init__(self, network: torch.nn.Module, outpath: str, do_reload=True):
        self.network = network
        self.outpath = Path(outpath)
        self.do_reload = do_reload
        self.checkpoints = []

    def __enter__(self):
        info_file = Path(f'{self.outpath.stem}.info')
        prefix = self.outpath.stem
        loaded = 0
        if self.do_reload and info_file.exists():
            with open(info_file,'r') as f:
                self.checkpoints = json.load(f)

            best_checkpoint = max(self.checkpoints)
            best_checkpoint_file = f'{prefix}-ep{best_checkpoint}.weights'
            print(f'Loading model from {best_checkpoint_file}')
            state = torch.load(best_checkpoint_file)
            self.network.load_state_dict(state)
            loaded = best_checkpoint
        return self.save_model, loaded

    def save_model(self, epoch: int):
        prefix = self.outpath.stem
        checkpoint_file = f'{prefix}-ep{epoch}.weights'
        print(f'Saving model at {checkpoint_file}')
        parent = Path(checkpoint_file).parent.absolute()

        if not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_file)
        self.checkpoints.append(epoch)
        info_file = Path(f'{self.outpath.stem}.info')
        with open(info_file,'w') as f:
            json.dump(self.checkpoints,f)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'Saving model at {self.outpath}')
        parent = Path(self.outpath).parent.absolute()
        if not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        torch.save(self.network.state_dict(), self.outpath)
