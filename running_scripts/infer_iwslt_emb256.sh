PREFIX=/

export CUDA_VISIBLE_DEVICES=${1:-0}

if [ $# != 5 ]; then
  echo "Please enter 'gpu' 'src', 'tgt', 'lang_pair', 'model' as input."
  exit
fi

SRC=$2
TGT=$3
LANG_PAIR=$4
MODEL=$5
DATA=iwslt14_${LANG_PAIR}
DATA_PATH=${PREFIX}/data/${DATA}/${DATA}/bin_data
CODE_PATH=${PREFIX}/fairseq_mix
BEAM=5
LENP=1.0
SENPIECE_MODEL_PATH=${PREFIX}/data/${DATA}/${DATA}/${LANG_PAIR}.model
nvidia-smi

MODEL_PATH=$PREFIX/models/$MODEL
python -c "import torch; print(torch.__version__)"
pip install sentencepiece --user

python $CODE_PATH/generate.py $DATA_PATH --source-lang $SRC --target-lang $TGT --path ${MODEL_PATH}/checkpoint_best.pt --senpiece-model $SENPIECE_MODEL_PATH --batch-size 128 --beam $BEAM --lenpen $LENP --quiet --remove-bpe --no-progress-bar | tee $PREFIX/translation/$MODEL.bleu

MOSES=$PREFIX/scripts/mosesdecoder/scripts
TOKENIZER=$MOSES/tokenizer/tokenizer.perl

echo "detokenize and calculate multi-bleu"
$TOKENIZER -l $TGT < $MODEL_PATH/ref_tgt.txt > $MODEL_PATH/ref_tgt.tok 
$TOKENIZER -l $TGT < $MODEL_PATH/trans.txt > $MODEL_PATH/trans.tok
$MOSES/generic/multi-bleu.perl $MODEL_PATH/ref_tgt.tok < $MODEL_PATH/trans.tok
