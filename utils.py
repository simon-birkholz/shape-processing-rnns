import math
import wandb

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

class WBContext:

    def __init__(self, config: Dict):
        
        self.suppress = config.get('wb_suppress') is not None
        config.pop('wb_suppress', None)
        self.params = config.copy()
        self.group = config.pop('wb_group', None)
        self.config = config

    def __enter__(self):
        if not self.suppress:
            self.run = wandb.init(
                project='shape-processing-rnns',
                entity='cenrypol',
                group=self.group,
                config=dict(params=self.params)
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.suppress:
            self.run.finish()

    def get_config() -> Dict:
        return self.config