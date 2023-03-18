#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.birkholz@uni-ulm.de

# Prepare the datasets and environment

WS_DATASET=$(ws_find cv_datasets)

echo $WS_DATASET

#conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shape-processing-rnns

cp -r $WS_DATASET $TMP/shape-proc-rnns

cd ~/shape-processing-rnns

#DATASET_PATH=$TMP

#envsubst < config.json.template

# Run the model training

python main.py config.json

# Cleanup
