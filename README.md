# Shape-Selective Processing in Recurrent Deep Neural Networks

This is the implementation belonging to my master's thesis.

## Dependencies and Installation

Several dependencies are needed to run the project. It is recommended to use conda with `environment.yml` to install most of the dependencies.
```
conda env create -f environment.yml
```
The training of the model uses `ffcv` to speed up the data loading of the ImageNet dataset. Under windows this needs additional installation requirements.
See [the github page](https://github.com/libffcv/ffcv) for additional informations.

## Usage

To train a model on a dataset you must provide a config similar to `config.json.template`.
```
python main.py config.json --out model.weights
```
The model will be trained with the according parameters and the weights will be saved in `model.weights` when completed.
Parameters of the config are as follows:
- TODO
- 
Additional Parameters could be:
- TODO
- 
## Experiments and Evaluation

For the analysis on diagnostic stimuli and the representational similarity analysis see `experiments`.


## Acknowledgements

This repo reuses code from:
- https://github.com/cJarvers/shapebias
