# Shape-Selective Processing in Recurrent Deep Neural Networks

This is the implementation belonging to my master's thesis.

## Dependencies and Installation

Several dependencies are needed to run the project. It is recommended to use conda with `environment.yml` to install most of the dependencies.
```
conda env create -f environment.yml
```
The training of the model uses `ffcv` to speed up the data loading of the ImageNet dataset. Under windows this needs additional installation requirements.
See [the github page](https://github.com/libffcv/ffcv) for additional informations.

Additionally, to use the adapted implementation for the `hgru` and `fgru` units, you need to install the code from the adapted [serrelabmodels](https://github.com/simon-birkholz/serrelabmodels) repository
```
git clone https://github.com/simon-birkholz/serrelabmodels.git
cd serrelabmodels
pip install -e .
```

## Usage

To train a model on a dataset you must provide a config similar to `config.json.template`.
```
python main.py config.json --out model.weights
```
The model will be trained with the according parameters and the weights will be saved in `model.weights` when completed.
Parameters of the config are as follows:
- `dataset`: The keyword for the dataset to be loaded
- `dataset_path`: File path to the directory where the data is located
- `dataset_val_path` (Optional) : File path to the validation data (only needed for some datasets)
- `batch_size`: The effective batch size with which the model will be trained
- `epochs`: Number of epochs the network will be trained. If `early-stop` is supplied as value, then the early stopping algorithm will be used.
- `learnng_rate` : Learning rate that will be used
- `model_base` : The overall base architecture of the model
- `tower_type` : defines the configuration of filter sizes that will be loaded
- `optimizer` : The optimizer that will be used
- `cell_type` : The type of recurrent cell, that is going to be used
- `time_steps` : Number of unrollment steps used on the recurrent cell
-
Additional Parameters could be:
- `lr_scheduler` : If specified a learning rate scheduler will be used
- `lr_step` : Step size of the learning rate scheduler
- `momentum` : Momentum of the optimizer (if optimizer uses momentum)
- `weight_decay` : Weight decay for the optimizer
- `normalization` : If specified the normalization layer that is going to be used
- `dropout` : Dropout factor of the spatial dropout layers
- `do_gradient_clipping` : If true, gradient clipping is applied
- `skip_first` : If true, the first layer does not contain a recurrent cell unit
- `do_preconv` : If true, a 1x1 convolution layer is applied before the recurrent cell unit
- `batch_max` : If the effective batch size is larger than the max batch size gradient accumulation is used to accommodate this
- 
Additional Parameters for the Weights and Biases wrapper (used during development):
- `wb_suppress` : If true, the wand api will not be used
- `wb_group` : The group name for the run on the wandb dashboard
- `wb_run_name` : The name of the run that will be displayed on the wandb dashboard
- `wb_sweep_id` : Used to run a hyperparameter search using the Sweep api of wandb

To use reload already existing weights the `ModelFileContext` class is used.
- `do_reload` : If true, this allows to reload previous runs that have been started with the same outfile parameter

This will save the model as well as the optimizer state every 5 epochs. It will also track the available files in a info file.
When a new run is started it will look for the highest checkpoint and tries to load it to contine training.

## Experiments and Evaluation

For the analysis on diagnostic stimuli and the representational similarity analysis see `experiments`.


## Acknowledgements

This repo reuses code from:
- https://github.com/cJarvers/shapebias

