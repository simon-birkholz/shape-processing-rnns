#!/bin/bash
start=`date +%s`

WS_DATASET=$(ws_find cv_datasets)

echo $WS_DATASET

#mkdir -p $TMP/anaconda3
cp -r ~/anaconda3 $TMP

#rsync -avhz ~/anaconda3 $TMP

#tar cf - ~/anaconda3 | (cd $TMP; tar xf -)

source $TMP/anaconda3/etc/profile.d/conda.sh
#source ~/anaconda3/etc/profile.d/conda.sh
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

end=`date +%s`
runtime=$(echo "$end - $start" | bc -l)

echo "Took $runtime seconds for preprocessing"

