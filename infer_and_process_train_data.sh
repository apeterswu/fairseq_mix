#!/usr/bin/env bash
echo "Translate Training data."
PREFIX=''

export CUDA_VISIBLE_DEVICES=${1:-0}

if [ $# != 5 ]; then
    echo "Please input 'gpu', 'src', 'tgt', 'lang_pair', 'model'."
    exit
fi

SRC=$2
TGT=$3
LANG_PAIR=$4
MODEL=$5   
DATA=iwslt14_${LANG_PAIR}
DATA_PATH=${PREFIX}/data/${DATA}/${DATA}/bin_data
BPECODE=${PREFIX}/data/${DATA}/${DATA}/bpe_code
CODE_PATH=${PREFIX}/fairseq_mix
BEAM=5
LENP=1.0
SENPIECE_MODEL_PATH=${PREFIX}/data/${DATA}/${DATA}/${LANG_PAIR}.model

MODEL_PATH=${PREFIX}/models/$MODEL
pip install sentencepiece --user

python $CODE_PATH/generate.py $DATA_PATH --source-lang $SRC --target-lang $TGT --path ${MODEL_PATH}/checkpoint_best.pt --senpiece-model $SENPIECE_MODEL_PATH --batch-size 128 --beam $BEAM --lenpen $LENP --quiet --remove-bpe --gen-subset train

MOSES=$PREFIX/scripts/mosesdecoder/scripts
TOKENIZER=$MOSES/tokenizer/tokenizer.perl
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl

if [ $LANG_PAIR == 'deen' ]; then
    BPEROOT=$PREFIX/scripts/subword-nmt/subword-nmt
else
    BPEROOT=$PREFIX/scripts/fastBPE/fast
fi

echo "1. Detokenize translated BPE and tokenize it to sentencepiece data."
$DETOKENIZER -l $SRC < $MODEL_PATH/bpe_src.tok > $MODEL_PATH/bpe_src.tok.detok
$DETOKENIZER -l $TGT < $MODEL_PATH/bpe_trans.tok > $MODEL_PATH/bpe_trans.tok.detok

python $PREFIX/scripts/spm_split_single.py $MODEL_PATH/bpe_src.tok.detok $SENPIECE_MODEL_PATH
python $PREFIX/scripts/spm_split_single.py $MODEL_PATH/bpe_trans.tok.detok $SENPIECE_MODEL_PATH
# $MODEL_PATH/bpe_src.tok.detok.sp
# $MODEL_PATH/bpe_trans.tok.detok.sp

echo "2. Tokenize sentencepiece data and tokenize it to BPE data."
$TOKENIZER -l $SRC -no-escape -threads 4 < $MODEL_PATH/sp_src.detok > $MODEL_PATH/sp_src.detok.tok
$TOKENIZER -l $TGT -no-escape -threads 4 < $MODEL_PATH/trans.txt > $MODEL_PATH/sp_trans.detok.tok

if [ $LANG_PAIR == 'deen' ];
then
    python $BPEROOT/apply_bpe.py -c $BPECODE < $MODEL_PATH/sp_src.detok.tok > $MODEL_PATH/sp_src.detok.tok.bpe
    python $BPEROOT/apply_bpe.py -c $BPECODE < $MODEL_PATH/sp_trans.detok.tok > $MODEL_PATH/sp_trans.detok.tok.bpe
else
    $BPEROOT applybpe $MODEL_PATH/sp_src.detok.tok.bpe $MODEL_PATH/sp_src.detok.tok $BPECODE
    $BPEROOT applybpe $MODEL_PATH/sp_trans.detok.tok.bpe $MODEL_PATH/sp_trans.detok.tok $BPECODE
fi

echo "3. Combine the original bilingual data with translated data and save it to ${DATA}/${DATA}_trans"
DATA_PATH_TRANS=${PREFIX}/data/${DATA}/iwslt14_${SRC}${TGT}_trans
mkdir -p $DATA_PATH_TRANS
cat ${PREFIX}/data/${DATA}/${DATA}/train.$SRC $MODEL_PATH/sp_src.detok.tok.bpe > ${DATA_PATH_TRANS}/train.$SRC
# wc -l ${PREFIX}/data/${DATA}/${DATA}_trans/train.$SRC
cat ${PREFIX}/data/${DATA}/${DATA}/train.$TGT $MODEL_PATH/sp_trans.detok.tok.bpe > ${DATA_PATH_TRANS}/train.$TGT
# wc -l ${PREFIX}/data/${DATA}/${DATA}_trans/train.$TGT

cat ${PREFIX}/data/${DATA}/${DATA}/train.$SRC.sen_piece $MODEL_PATH/bpe_src.tok.detok.sp > ${DATA_PATH_TRANS}/train.$SRC.sen_piece
# wc -l ${PREFIX}/data/${DATA}/${DATA}_trans/train.$SRC.sen_piece
cat ${PREFIX}/data/${DATA}/${DATA}/train.$TGT.sen_piece $MODEL_PATH/bpe_trans.tok.detok.sp > ${DATA_PATH_TRANS}/train.$TGT.sen_piece
# wc -l ${PREFIX}/data/${DATA}/${DATA}_trans/train.$TGT.sen_piece

cp ${PREFIX}/data/${DATA}/${DATA}/valid.* ${DATA_PATH_TRANS}/
cp ${PREFIX}/data/${DATA}/${DATA}/test.* ${DATA_PATH_TRANS}/

echo "4. Binarize the data with previously generated vocabulary."
python $CODE_PATH/preprocess.py  --source-lang $SRC --target-lang $TGT \
--trainpref $DATA_PATH_TRANS/train --validpref $DATA_PATH_TRANS/valid --testpref $DATA_PATH_TRANS/test \
--destdir $DATA_PATH_TRANS/bin_data \
--srcdict $DATA_PATH/dict.$SRC.txt --tgtdict $DATA_PATH/dict.$TGT.txt \
--srcdict-sen-piece $DATA_PATH/dict.$SRC.sen_piece.txt --tgtdict-sen-piece $DATA_PATH/dict.$TGT.sen_piece.txt \
--workers 20

echo "Finished binarizing data in ${DATA_PATH_TRANS}/bin_data"


