MODEL=transformer
ARCH=transformer_dec_one_iwslt_de_en
PREFIX=/

export CUDA_VISIBLE_DEVICES=${1:-0}

INPUT1=$2
INPUT2=$3
DIRC=$4
DATA=iwslt14_${INPUT1}${INPUT2}
LR=0.0005
DATA_PATH=${PREFIX}/data/${DATA}/${DATA}/bin_data
CODE_PATH=${PREFIX}/fairseq_mix
nvidia-smi

if [ $DIRC == "src2tgt" ]; then
  SRC=$INPUT1
  TGT=$INPUT2
else  # tgt2src 
  SRC=$INPUT2
  TGT=$INPUT1
fi

MODEL_PATH=$PREFIX/models/iwslt14_${SRC}${TGT}_mix_dec_one_emb256
mkdir -p ${MODEL_PATH}

python -c "import torch; print(torch.__version__)"
python $CODE_PATH/train.py $DATA_PATH 	--arch $ARCH --share-all-embeddings \
  --source-lang $SRC --target-lang $TGT  --optimizer adam --adam-betas "(0.9, 0.98)" \
  --clip-norm 0.0  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
  --warmup-updates 4000  --lr $LR --min-lr 1e-09 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --save-interval-updates 0 --max-update 120000 --keep-interval-updates 100 \
  --dropout 0.3 --max-tokens 4096 \
  --enc-drop-path-ratio 0.2  --dec-drop-path-ratio 0.3 \
  --encoder-embed-dim 256 --decoder-embed-dim 256 \
  --save-dir $MODEL_PATH  --seed 1 --restore-file checkpoint_best.pt  --update-freq 1 | tee $PREFIX/scripts/logs/iwslt14_${SRC}${TGT}_mix_dec_one_emb256.log
