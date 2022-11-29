#!/bin/bash

# path to NER training and validation data
NER_DIR=../DDI_data/NER_data

# path to BioBERT model
BERT_DIR=../biobert_large_hybrid

# path for output
mkdir ../NERoutput
OUTPUT_DIR=../NERoutput

# run named entity recognition (NER) training and validation script
python3 myrun_ner.py \
 --data_dir=$NER_DIR \
 --bert_model=$BERT_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_batch_size=32 \
 --max_seq_length=128 \
 --do_train \
 --num_train_epochs=50 \
 --multi_gpu \
 --do_eval \
 --eval_on=dev
 
# run de-tokenization script
python3 ./biocodes/myner_detokenize.py \
 --token_test_path=$OUTPUT_DIR/token_test.txt \
 --label_test_path=$OUTPUT_DIR/label_test.txt \
 --tokenized_answer_path=$OUTPUT_DIR/token_label_test.tsv \
 --answer_path=$NER_DIR/valid.txt \
 --output_dir=$OUTPUT_DIR
  
