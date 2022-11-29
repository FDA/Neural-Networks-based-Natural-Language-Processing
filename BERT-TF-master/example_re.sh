#!/bin/bash

# path to RE training and validation data
RE_DIR=../DDI_data/RE_data

# path to BioBERT model
BERT_DIR=../biobert_large_hybrid

# path for output 
mkdir ../REoutput
OUTPUT_DIR=../REoutput

# run relation extraction (RE) training and validation script
python3 myrun_re.py \
 --data_dir=$RE_DIR \
 --bert_model=$BERT_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_batch_size=32 \
 --max_seq_length=128 
 --do_train 
 --num_train_epochs=2 
 --multi_gpu \
 --do_eval \
 --eval_on=dev

