import torch.optim as optim

OPTIMIZERS = {
    'adam': {
        'fn': lambda *args, **kwargs: optim.Adam(*args, **kwargs),
        'args': {
            'lr': None,
            'weight_decay': None
        }
    },
    'adamw': {
        'fn': lambda *args, **kwargs: optim.AdamW(*args, **kwargs),
        'args': {
            'lr': None
        }
    },
    'sgd': {
        'fn': lambda *args, **kwargs: optim.SGD(*args, **kwargs),
        'args': {
            'lr': None,
            'weight_decay': None,
            'momentum': None,
            'nesterov': True
        }
    },
    'rms': {
        'fn': lambda *args, **kwargs: optim.RMSprop(*args, **kwargs),
        'args': {
            'lr': None,
            'weight_decay': None,
            'momentum': None
        }
    }
}

LR_SCHEDULER = {
    'step' : {
        'fn' : lambda *args, **kwargs: optim.lr_scheduler.StepLR(*args, **kwargs),
        'args' : {
            'step_size' : 30,
            'gamma' : 0.1
        }
    }
}


def get_argument_instance(arguments, key, *args, is_optional=False, **kwargs):
    if key and key in arguments.keys():
        args = list(args)
        arg_instance = arguments[key]
        needed_kwargs = {k: v for k, v in kwargs.items() if k in arg_instance['args'] and v is not None}
        default_args = {k: v for k, v in arg_instance['args'].items() if v is not None}
        tt_args = default_args | needed_kwargs
        result = arg_instance['fn'](*args, **tt_args)
        print(f' Selected {key} with arguments: {tt_args}')
        return result
    else:
        # unkown key
        if is_optional:
            return None
        else:
            raise ValueError(f'Missing argument. Cannot find value {key}')
