# Natural Language Processing to Identify Directional Pharmacokinetic Drug-Drug Interaction (DDI)

A BioBERT-based NLP model to perform relation extraction (RE) and named entity recognition (NER) to identify directional DDIs.  Model weights are from [BioBERT-Large v1.1](https://github.com/dmis-lab/biobert).  Training and validation scripts are modified from [https://github.com/kamalkraj/BERT-NER-TF](https://github.com/kamalkraj/BERT-NER-TF) and implemented with TensorFlow v2.

## Requirements

- `python3`
- `tensorflow` (version >= 2.0)
- `fastprogress` (version >= 0.1.21)
- `seqeval` (version >= 0.0.5)

## Sub-directories

- `BERT-TF-master` contains the main scripts for training and validation
- `DDI_data` contains the training and validation datasets for RE and NER steps

## Usage

- Download [BioBERT-Large v1.1](http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_large_v1.1_pubmed.tar.gz).  Save as a sub-directory, e.g. 'biobert_large'.
- Convert TensorFlow version 1 model weights to TensorFlow version 2 model weights; follow procedure in `tf1_convert_tf2.sh`.
- First, run training and validation for RE step.  To do this, run `myrun_re.py` under `BERT-TF-master` directory.  An example bash script, `example_re.sh`, shows the various command line arguments supplied to `myrun_re.py`.
- Second, run training and validation for NER step.  To do this, run `myrun_ner.py` under `BERT-TF-master` directory, followed by `myner_detokenize.py` under `BERT-TF-master/biocodes` directory.  An example bash script, `example_ner.sh`, shows the various command line arguments supplied to both these scripts.

## Disclaimer

This software and documentation were developed by the authors in their capacities as Oak Ridge Institute for Science and Education (ORISE) research fellows at the U.S. Food and Drug Administration (FDA).

FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, FDA makes no representations that the use of the Software will not infringe any patent or proprietary rights of third parties. The use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions.
