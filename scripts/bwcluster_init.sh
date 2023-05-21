#!/bin/bash

cd ~/shape-processing-rnns

RAND_ID=$(echo $RANDOM | md5sum | head -c 20)
echo $RAND_ID

# make temporary copy of config file

mkdir -p tmp

CONF_IN=tmp/config-$RAND_ID.json.in
CONF_READY=tmp/config-$RAND_ID.json
JOB=tmp/job-$RAND_ID.sh
OUT=tmp/slurm-$RAND_ID.out
WEIGHTS=tmp/model-$RAND_ID.weights

nojobs=${1:-1}
NAME="${2:none}"

cp config.json.in $CONF_IN

touch $JOB

cat scripts/preamble.sh >> $JOB
echo "#SBATCH --output=$OUT" >> $JOB

cat scripts/bwcluster_pre.sh >> $JOB

echo "cat $CONF_IN | envsubst > $CONF_READY" >> $JOB

echo "python main.py $CONF_READY --out $WEIGHTS --wbname $NAME" >> $JOB

loop_cnt=1

while [ ${loop_cnt} -le ${nojobs} ] ; do
    sbatch -p gpu_4_a100 -t 48:00:00 $JOB
    let loop_cnt+=1
done

