#!/bin/sh
#$ -cwd
#$ -P CDERID0047
#$ -l h_rt=12:00:00
#$ -l h_vmem=100G
#$ -S /bin/sh
#$ -j y
#$ -N data_array

#$ -pe orte1 1
#$ -t 1-9

#$ -l gpus=2                 #each job this many GPUs
#$ -l ncores=1                  #seems that only 1 CPU core is needed per job

#$ -o arrayout

export CUDA_VISIBLE_DEVICES="${SGE_GPU}"
export NVIDIA_VISIBLE_DEVICES="${SGE_GPU}"

echo "Running $SGE_TASK_ID of job $JOB_NAME, $JOB_ID on $HOSTNAME"

source /projects/mikem/applications/centos7/ananconda3/set-env.sh
source /projects/mikem/opt/cuda/set_env.sh
export LD_LIBRARY_PATH=/app/GPU/CUDA-Toolkit/cuda-11.1/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/lizhi/.local/lib/python3.7:$PYTHONPATH

# debug
echo "SGE_GPU=$SGE_GPU" 
echo "SGE_BINDING=$SGE_BINDING"

# export allocated GPUs
export DEV=$(echo $SGE_GPU | awk -F',' '{printf $1}' )

# export allocated CPU
export CPU_DEV1=$(echo $SGE_BINDING | awk -F' ' '{printf $1}' )

# debug
echo "SGE_GPU=$SGE_GPU" # test
echo "DEV=$DEV" # test

# SGE_GPU variable contains a spac -delimited device IDs, such as 0 or 0 1 2 
# depending on the number of gpu resources to be requested. 
# Use the device ID for cudaSetDevice().

APP=./array_big_tf_GPU.sh

time $APP
