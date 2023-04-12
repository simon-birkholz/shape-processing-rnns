import math
import wandb
import functools
from typing import Dict


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
                        config=dict(params=self.params)
                    )
                    updated_values = dict(self.config,**self.run.config)
                    updated_values.pop('params',None)
                    train_fun(**updated_values)
                def agent_wrapper(train_fun):
                    wandb.agent(sweep_id=complete_sweep_id, function=functools.partial(train_fun_wrapper,train_fun))
                return agent_wrapper
            else:
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    group=self.group,
                    config=dict(params=self.params)
                )
                return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.suppress:
            self.run.finish()
