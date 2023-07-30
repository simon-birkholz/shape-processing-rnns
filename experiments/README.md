# Experiments

This folder contains the scripts necessary to reproduce the results to the diagnostic stimuli analysis and the representational similarity analysis of the thesis.

## Usage

To reproduce run either `diagnostic_stimuli.py` or `rs_analysis.py`.
Example:
```
python diagnostic_stimuli.py --datasets normal foreground shilouette frankenstein serrated --path /to/ds --weights model.weights -cell_type fgru --norm layernorm --time_steps 15 --out out.file
```

# Visualization

The plots are generated using the following files:
- `plot_diagnostic_images.py`
- `plot_diagnostic_stimuli.py`
- `plot_diagnostic_stimuli_timeseries.py`
- `plot_imagenet.py`
- `plot_model_complexity_acurracy.py`
- `plot_rs_analysis.py`
- `timeseries_rs_analysis.py`

The parameters for the figures used in the thesis can be found in:
- `plot_final.sh`
- `plot_time_rsa.sh`

# Other files

Other files are:
- `ds_transforms.py`: Contains the data transformations to generate the diagnostic stimuli
- `fdr.py`: Contains the code for the benjamini hochberg procedure
- `imagenet_val_set.py`: Calculates the accuracy on the ImageNet validation split