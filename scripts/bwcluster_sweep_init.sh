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

cp config.json.in $CONF_IN

NAME="${2:none}"

touch $JOB

cat scripts/preamble.sh >> $JOB
echo "#SBATCH --output=$OUT" >> $JOB

cat scripts/bwcluster_pre.sh >> $JOB

echo "cat $CONF_IN | envsubst > $CONF_READY" >> $JOB

echo "python main.py $CONF_READY --out $WEIGHTS --wbname $NAME" >> $JOB

nojobs=${1:-5}
dep_type="afterany"

loop_cnt=1

# taken from https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm#Chain_jobs
while [ ${loop_cnt} -le ${nojobs} ] ; do

	if [ ${loop_cnt} -eq 1 ] ; then
		slurm_opt=""
	else
		slurm_opt="-d ${dep_type}:${jobID}"
	fi

	jobID=$(sbatch -p gpu_4_a100 ${slurm_opt} -t 48:00:00 $JOB 2>&1 | sed 's/[S,a-z]* //g')
	
	if [[ "${jobID}" =~ "*sbatch::error*" ]] ; then
      		echo "   -> submission failed!" ; exit 1
   	else
      		echo "   -> job number = ${jobID}"
   	fi
	let loop_cnt+=1
done
