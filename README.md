# Sequence Generation with Mixed Representations
This repository contains the code of "Sequence Generation with Mixed Representations" - ICML2020. 
The project is based on the [fairseq (version 0.6.1)](https://github.com/pytorch/fairseq/tree/v0.6.1).

```
@inproceedings{wu2019depth,
  title={Sequence Generation with Mixed Representations},
  author={Wu, Lijun and Xie, Shufang and Xia, Yingce and Fan, Yang and Qin, Tao and Lai, Jianhuang and Liu, Tie-Yan},
  booktitle={ICML 2020},
  year={2020}
}
```

# Requirements and Installation
* A [PyTorch installation (0.4.0)](http://pytorch.org/)
and install fairseq with:
```
pip install -r ./fairseq_mix/requirements.txt
python ./fairseq_mix/setup.py build develop
```
* install [FastBPE](https://github.com/glample/fastBPE)
* install [mosesdecoder](https://github.com/moses-smt/mosesdecoder)


# Preprocess the data:
1. Prepare BPE data, it should be same as the [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh)
2. Prepare SentencePiece (SP) data, learning SP vocabulayr and tokenized data by running:
``` sen_piece_learn.py ```
3. Preprocess the BPE and SP data by running:
``` preprocess.py ```

# Training models:
1. Pre-train mix_presentation model
``` ruuning_scripts/train_mix_iwslt14_emb256.sh ```
2. Generate the translation data for self-training (co-teaching data generation)
``` infer_and_process_train_data.sh ```
3. Self-training models (co-teaching model training)
``` ruuning_scripts/train_mix_iwslt14_emb256_reset_trans.sh ```

# Inference:
1. Infer pre-train model (mix_representation model)
``` ruuning_scripts/infer_iwslt_emb256.sh ```
2. Infer self-training model (co-teaching model)
``` running_scripts/infer_iwslt_emb256_reset_trans.sh ```





