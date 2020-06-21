import os
import sys
import argparse
import sentencepiece as spm
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='source language')
parser.add_argument('--tgt', help='target language')
parser.add_argument('--vocab_size', type=int, help='vocabulary size')
args = parser.args()

data_prefix = ''
train_data = ''

# train the sentencepiece model
spm.SentencePieceTrainer.Train('--input={} --model_prefix={}2{} --vocab_size={} --shuffle_input_sentence=true'.format(train_data, args.src, args.tgt, args.vocab_size))

model = '{}/{}2{}.model'.format(data_prefix, args.src, args.tgt)
vocab = '{}/{}2{}.vocab'.format(data_prefix, args.src, args.tgt)

sp = spm.SentencePieceProcessor()
sp.Load(model)
for sub_set in ['train', 'valid', 'test']:
    for lang in [args.src, args.tgt]:
        sub_lang = '{}/{}.{}.detok'.format(data_prefix, sub_set, lang)
        sub_lang_pro = '{}/{}.{}.sen_piece'.format(data_prefix, sub_set, lang)
        fin = open(sub_lang, 'r', encoding='utf-8')
        fout = open(sub_lang_pro, 'w', encoding='utf-8')
        contents = fin.readlines()
        for line in contents:
            sp_sen_piece = sp.EncodeAsPieces(line)
            fout.write(' '.join(sp_sen_piece))
            fout.write('\n')
        fin.close()
        fout.close()





