MODEL=transformer
ARCH=transformer_dec_one_iwslt_de_en
PREFIX=/

export CUDA_VISIBLE_DEVICES=${1:-0}

SRC=$2
TGT=$3
LANG=$4
WARM_PATH=$5
WARM_MODEL=${PREFIX}/models/${WARM_PATH}/checkpoint_best.pt
DATA=iwslt14_${LANG}
DATA_TRANS=iwslt14_${SRC}${TGT}_trans  
LR=0.0005
DATA_PATH=${PREFIX}/data/${DATA}/${DATA_TRANS}/bin_data
CODE_PATH=${PREFIX}/fairseq_mix
nvidia-smi

MODEL_PATH=$PREFIX/models/${DATA_TRANS}_mix_dec_one_emb256_reset
mkdir -p ${MODEL_PATH}


if ! [ -f ${MODEL_PATH}/checkpoint_best.pt ]; then
  echo copy warm model ${WARM_MODEL} to the model path ${MODEL_PATH}
  cp ${WARM_MODEL} ${MODEL_PATH}/checkpoint_best.pt
fi


python -c "import torch; print(torch.__version__)"
python $CODE_PATH/train.py $DATA_PATH 	--arch $ARCH --share-all-embeddings \
  --source-lang $SRC --target-lang $TGT  --optimizer adam --adam-betas "(0.9, 0.98)" \
  --clip-norm 0.0  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
  --warmup-updates 4000  --lr $LR --min-lr 1e-09 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --save-interval-updates 0 --max-update 150000 --keep-interval-updates 100 \
  --dropout 0.3 --max-tokens 4096 \
  --enc-drop-path-ratio 0.2  --dec-drop-path-ratio 0.3 \
  --encoder-embed-dim 256 --decoder-embed-dim 256 \
  --reset-lr-scheduler --reset-optimizer \
  --save-dir $MODEL_PATH  --seed 1 --restore-file checkpoint_best.pt  --update-freq 1 | tee $PREFIX/scripts/logs/${DATA_TRANS}_mix_dec_one_emb256_reset.log
