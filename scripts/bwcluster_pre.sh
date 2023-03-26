#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.birkholz@uni-ulm.de
#SBATCH --gres=gpu:1
# Prepare the datasets and environment

start=`date +%s`

WS_DATASET=$(ws_find cv_datasets)

echo $WS_DATASET

#conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate shape-processing-rnns

echo "Copying datasets"

mkdir -p $TMP/shape-proc-rnns
echo $TMP/shape-proc-rnns
#cp -r $WS_DATASET $TMP/shape-proc-rnns

#unzip -q $WS_DATASET/imagenet-object-localization-challenge.zip -d $TMP/shape-proc-rnns
#cp $WS_DATASET/imagenet_class_index.json $TMP/shape-proc-rnns
#cp $WS_DATASET/ILSVRC2012_val_labels.json $TMP/shape-proc-rnns

cp $WS_DATASET/imagenet_kaggle_train.beton $TMP/shape-proc-rnns
cp $WS_DATASET/imagenet_kaggle_val.beton $TMP/shape-proc-rnns

#rsync -ah $WS_DATASET $TMP/shape-proc-rnns

cd ~/shape-processing-rnns

#DATASET_PATH=$TMP

cat config.json.in | envsubst > config.json

end=`date +%s`
runtime=$(echo "$end - $start" | bc -l)

echo "Took $runtime seconds for preprocessing"

# Run the model training

python main.py config.json

# Cleanup
