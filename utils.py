import math


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
