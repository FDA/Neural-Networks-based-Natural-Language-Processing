#!/bin/sh

# example procedure to convert TensorFlow version 1 model weights to TensorFlow version 2 model weights

# path to BioBERT-Large v1.1 checkpoint file as downloaded from:
# https://github.com/dmis-lab/biobert
INPUT_CHECKPOINT=biobert_large/bio_bert_large_1000k.ckpt

# path to output directory
OUTPUT_DIR=biobert_large_converter2.1

# python script is originally sourced from:
# https://github.com/tensorflow/models/tree/r2.1_model_reference/official/nlp/bert
python3 tf1_to_keras_checkpoint_converter.py \
  --checkpoint_from_path=$INPUT_CHECKPOINT \
  --checkpoint_to_path=$OUTPUT_DIR \

# create directory for TensorFlow version 2 model weights
mkdir biobert_large_hybrid

# copy files from OUTPUT_DIR
cp biobert_large_converter2.1/biobert_large_converter2.1.data-00000-of-00001 biobert_large_hybrid/be
rt_model.ckpt.data-00000-of-00001

cp biobert_large_converter2.1/biobert_large_converter2.1.index biobert_large_hybrid/bert_model.ckpt.
index

# copy files from BioBERT-Large v1.1
cp biobert_large/bert_config_bio_58k_large.json biobert_large_hybrid/bert_config.jason

cp biobert_large/ vocab_cased_pubmed_pmc_30k.txt biobert_large_hybrid/vocab.txt
