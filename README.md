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
* Prepare BPE data, refer to [data_prepare](https://github.com/apeterswu/fairseq_mix/tree/master/examples/translation)

* Prepare SentencePiece (SP) data, learning SP vocabulayr and tokenized data by running:

``` 
python sen_piece_learn.py
```

* Preprocess the BPE and SP data by running:

``` 
python preprocess.py 
```


# Training models:
* Pre-train mix_presentation model

``` 
python ruuning_scripts/train_mix_iwslt14_emb256.sh 
```

* Generate the translation data for self-training (co-teaching data generation)

``` 
python infer_and_process_train_data.sh 
```

* Self-training models (co-teaching model training)
``` 
python ruuning_scripts/train_mix_iwslt14_emb256_reset_trans.sh 
```


# Inference:
1. Infer pre-train model (mix_representation model)

``` 
python ruuning_scripts/infer_iwslt_emb256.sh 
```

2. Infer self-training model (co-teaching model)

```
python running_scripts/infer_iwslt_emb256_reset_trans.sh 
```







