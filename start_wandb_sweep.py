import argparse
import wandb
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_config_file', type=str, help='Config file for wand sweep')

    args = parser.parse_args()

    if not os.path.exists(args.sweep_config_file):
        raise ValueError(f"Config file {args.sweep_config_file} not found")
    with open(args.sweep_config_file) as f:
        config = json.load(f)

    sweep_id = wandb.sweep(sweep=config, project='shape-processing-rnns', entity='cenrypol')

    print(sweep_id)
