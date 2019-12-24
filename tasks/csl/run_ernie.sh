#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl" 
MODEL_NAME="baidu_ernie"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export ERNIE_DIR=$PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/../../LGECdataset

# check python package 
check_bert4keras=`pip show bert4keras | grep "Version"`

if [ ! -n "$check_bert4keras" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git@v0.2.6
else
  echo "bert4keras installed."
fi

check_rouge=`pip show rouge | grep "Version"`

if [ ! -n "$check_rouge" ]; then
  pip install rouge
else
  echo "rouge installed."
fi

# check dataset

cd $GLUE_DATA_DIR/$TASK_NAME
if [ ! -f "train.tsv" ] || [ ! -f "val.tsv" ] ; then
  echo "Data does not exist."
else
  echo "Dataset exists."
fi

# download model
if [ ! -d $ERNIE_DIR ]; then
  mkdir -p $ERNIE_DIR
  echo "makedir $ERNIE_DIR"
fi
cd $ERNIE_DIR
if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
  rm *
  wget https://storage.googleapis.com/chineseglue/pretrain_models/baidu_ernie.zip
  unzip baidu_ernie.zip
  rm baidu_ernie.zip
else
  echo "model exists"
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../summary_baseline.py \
    --dict_path=$ERNIE_DIR/vocab.txt \
    --config_path=$ERNIE_DIR/bert_config.json \
    --checkpoint_path=$ERNIE_DIR/bert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.tsv \
    --albert=False \
    --epochs=3 \
    --batch_size=8 \
    --lr=1e-5 \
    --topk=1 \
    --max_input_len=256 \
    --max_output_len=32
