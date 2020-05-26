#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="afqmc" 
MODEL_NAME="chinese_roberta_wwm_large_ext_L-24_H-1024_A-16"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export ROBERTA_WWM_LARGE_DIR=$PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/../../CLGEdataset

# check python package 
check_bert4keras=`pip show bert4keras | grep "Version"`

if [ ! -n "$check_bert4keras" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git
else
  echo "bert4keras installed."
fi


# download model
if [ ! -d $ROBERTA_WWM_LARGE_DIR ]; then
  mkdir -p $ROBERTA_WWM_LARGE_DIR
  echo "makedir $ROBERTA_WWM_LARGE_DIR"
fi
cd $ROBERTA_WWM_LARGE_DIR
if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
  rm *
  wget -c https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
  unzip chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
  rm chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
else
  echo "model exists"
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../data_augmented_baseline.py \
    --dict_path=$ROBERTA_WWM_LARGE_DIR/vocab.txt \
    --config_path=$ROBERTA_WWM_LARGE_DIR/bert_config.json \
    --checkpoint_path=$ROBERTA_WWM_LARGE_DIR/bert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --epochs=100 \
    --batch_size=48 \
    --lr=1e-5 \
    --topk1=25 \
    --topk2=4 \
    --max_seq_len=128
