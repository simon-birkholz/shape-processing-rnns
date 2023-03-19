#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.birkholz@uni-ulm.de

# Prepare the datasets and environment

start=`date +%s`

WS_DATASET=$(ws_find cv_datasets)

echo $WS_DATASET

#conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shape-processing-rnns

echo "Copying datasets"

#cp -r $WS_DATASET $TMP/shape-proc-rnns

cd ~/shape-processing-rnns

#DATASET_PATH=$TMP

#envsubst < config.json.template

end=`date +%s`
runtime=$(echo "$end - $start" | bc -l)

echo "Took $runtime seconds for preprocessing"

# Run the model training

python main.py config.json

# Cleanup
