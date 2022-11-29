#!/bin/sh

echo check for gpu: nvidia-smi output:
nvidia-smi
echo

# Get start of job information
START_TIME=`date +%s`

#WORKERDIR=/scratch/joel.zirkle/DDI/github_scripts/REoutput/run_${JOBID}/${JOB_NAME}_${SGE_TASK_ID}
WORKERDIR=/scratch/joel.zirkle/DDI/github_scripts/NERoutput/run_${JOBID}/${JOB_NAME}_${SGE_TASK_ID}
mkdir -p ${WORKERDIR}

export BERT_DIR=/scratch/joel.zirkle/DDI/github_scripts/biobert_large_hybrid
export BERT_VOCAB=$BERT_DIR/vocab.txt
export BERT_CONFIG=$BERT_DIR/bert_config.json
export BERT_INIT=$BERT_DIR/bert_model.ckpt

export RE_DIR=/scratch/joel.zirkle/DDI/github_scripts/DDI_data/RE_data
export TASK_NAME=MRPC
export DATA_DIR=/scratch/joel.zirkle/DDI/github_scripts/DDI_data/NER_data
#export EPOCH=2       
export EPOCH=50               #note that NER needs more epochs          
export TRAIN_BATCH_SIZE=32

export OUTPUT_DIR=$WORKERDIR

#below is doing Relation Extraction
#python3 myrun_re.py \
# --data_dir=$RE_DIR --bert_model=$BERT_DIR --output_dir=$OUTPUT_DIR --train_batch_size=$TRAIN_BATCH_SIZE \
# --max_seq_length=128 --do_train --num_train_epochs=$EPOCH --multi_gpu --do_eval --eval_on=dev


#below is using my myrun_ner.py to output more files for entity-level performance
python3 myrun_ner.py \
 --data_dir=$DATA_DIR --bert_model=$BERT_DIR --output_dir=$OUTPUT_DIR --train_batch_size=$TRAIN_BATCH_SIZE \
 --max_seq_length=128 --do_train --num_train_epochs=$EPOCH --multi_gpu --do_eval --eval_on=dev

#below is doing de-tokenization; note that when mode=test the input file is valid.txt
#python3 ./biocodes/myner_detokenize.py \
#  --token_test_path=$OUTPUT_DIR/token_test.txt \
#  --label_test_path=$OUTPUT_DIR/label_test.txt \
#  --tokenized_answer_path=$OUTPUT_DIR/token_label_test.tsv \
#  --answer_path=$DATA_DIR/valid.txt \
#  --output_dir=$OUTPUT_DIR

# Get end of job information
EXIT_STATUS=$?
END_TIME=`date +%s`
