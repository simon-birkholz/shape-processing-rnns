import wandb
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import re
import collections.abc as abc

api = None
project = None


def to_dict(config_str: str):
    return json.loads(config_str)


def init_wandb_api():
    global api, project
    api = wandb.Api()

    project = api.project("shape-processing-rnns", entity="cenrypol")


def get_sweep_by_name(name: str):
    all_sweeps = project.sweeps()
    for sweep in all_sweeps:
        if sweep.name == name:
            return api.sweep('/'.join(sweep.path))
    return None


def get_runs_by_regex(regex: str, group: str = None):
    all_runs = api.runs('/'.join(project.path),
                        filters={'group': group} if group else {},
                        include_sweeps=False)
    return [r for r in all_runs if re.match(regex, r.name)]


def traverse_obj(obj, *keys):
    for k in keys:
        if isinstance(obj, abc.Mapping) and k in obj.keys():
            obj = obj[k]
        else:
            return None
    return obj


def run_to_label(run, attr) -> str:
    config_dict = to_dict(run.json_config)
    val = traverse_obj(config_dict, attr, 'value')
    if not val:
        val = traverse_obj(config_dict, 'params', 'value', attr)
    return f"{attr}: {val}"


def default_sort(run):
    return run.summary.get('val_acc', 0)


def plot_runs(runs, label_attr='learning_rate', show_top=1.0):
    ax = plt.gca()

    sorted_runs = sorted(runs, key=default_sort, reverse=True)
    keep_runs = sorted_runs[:int(len(sorted_runs) * show_top)]
    keep_data = [(run, history) for run in keep_runs if len(history := run.history()) > 0]

    for run, history in keep_data:
        history.plot(x='_step', y='val_acc', kind='line', ax=ax)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Accuracy')
    legend_labels = [run_to_label(r, label_attr) for r, _ in keep_data]
    ax.legend(legend_labels)

    plt.show()


def plot_runs_attr(runs,
                   ax,
                   attr: Tuple[str, str],
                   label_attr='learning_rate',
                   max_val=1.0,
                   min_val=0.0,
                   show_top=1.0):
    sorted_runs = sorted(runs, key=default_sort, reverse=True)
    keep_runs = sorted_runs[:int(len(sorted_runs) * show_top)]
    keep_data = [(run, history) for run in keep_runs if len(history := run.history()) > 0]

    for run, history in keep_data:
        history.plot(x='_step', y=attr[0], kind='line', ax=ax)

    ax.set_xlabel('Epochs')
    ax.set_ylabel(attr[1])
    ax.set_ylim([min_val, max_val])
    legend_labels = [run_to_label(r, label_attr) for r, _ in keep_data]
    ax.legend(legend_labels)


def get_max_value(runs, attr):
    keep_data = [history for run in runs if len(history := run.history()) > 0]
    return max([df.max()[attr] for df in keep_data])


def get_min_value(runs, attr):
    keep_data = [history for run in runs if len(history := run.history()) > 0]
    return min([df.min()[attr] for df in keep_data])


def plot_runs_accuracy(runs, title, label_attr='learning_rate', show_top=1.0):
    sorted_runs = sorted(runs, key=default_sort, reverse=True)
    keep_runs = sorted_runs[:int(len(sorted_runs) * show_top)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.suptitle(title)

    max_val = max(get_max_value(keep_runs, 'train_acc'), get_max_value(keep_runs, 'val_acc'))

    plot_runs_attr(keep_runs, ax1, ('val_acc', 'Validation Accuracy'), label_attr, max_val=max_val)

    plot_runs_attr(keep_runs, ax2, ('train_acc', 'Training Accuracy'), label_attr, max_val=max_val)

    plt.show()


def plot_runs_loss(runs, title, label_attr='learning_rate', show_top=1.0):
    sorted_runs = sorted(runs, key=default_sort, reverse=True)
    keep_runs = sorted_runs[:int(len(sorted_runs) * show_top)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.suptitle(title)

    max_val = max(get_max_value(keep_runs, 'training_loss'), get_max_value(keep_runs, 'val_loss'))
    min_val = min(get_min_value(keep_runs, 'training_loss'), get_min_value(keep_runs, 'val_loss'))

    plot_runs_attr(keep_runs, ax1, ('val_loss', 'Validation Loss'), label_attr, max_val=max_val, min_val=min_val)

    plot_runs_attr(keep_runs, ax2, ('training_loss', 'Training Loss'), label_attr, max_val=max_val, min_val=min_val)

    plt.show()


def get_metadata(run):
    system_data = run.history(stream='system')

    avg_power_watts = system_data.mean()['system.gpu.0.powerWatts']
    runtime = run.summary['_runtime']

    metadata = \
        {
            'display_name': run.name,
            'avg_power_watts': avg_power_watts,
            'runtime': runtime
        }
    return metadata


def get_power_consumption(runs):
    power_consumtion = [((mdata := get_metadata(r))['runtime'] / 3600, mdata['avg_power_watts']) for r in runs]
    total_wh = sum([t * w for t, w in power_consumtion])
    return total_wh


if __name__ == '__main__':
    init_wandb_api()

    GROUP = 'kw14-test-functionality'
    # test_sweep = get_sweep_by_name('fftower-conv')
    runs = get_runs_by_regex('f', group=GROUP)

    total_pwr = get_power_consumption(runs)
    print(f'Total Power Consumed: {total_pwr / 1000} (kWh)')
