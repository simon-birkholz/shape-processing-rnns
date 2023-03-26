#!/bin/bash

cd ~/shape-processing-rnns

RAND_ID=$(echo $RANDOM | md5sum | head -c 20)
echo $RAND_ID

# make temporary copy of config file

mkdir -p tmp

CONF_IN=tmp/config-$RAND_ID.json.in
CONF_READY=tmp/config-$RAND_ID.json
JOB=tmp/job-$RAND_ID.sh

cp config.json.in $CONF_IN

touch $JOB

cat scripts/preamble.sh >> $JOB

cat scripts/bwcluster_pre.sh >> $JOB

echo "cat $CONF_IN | envsubst > $CONF_READY" >> $JOB

echo "python main.py $CONF_READY" >> $JOB

#sbatch -p gpu_4_a100 -t 24:00:00 $JOB

