#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="webqa"
MODEL_NAME="chinese_roberta_wwm_large_ext_L-24_H-1024_A-16"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export ROBERTA_WWM_LARGE_DIR=$PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/../../CLGEdataset

# check python package 
check_bert4keras=`pip show bert4keras | grep "Version"`

if [ ! -n "$check_bert4keras" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git@v0.3.6
else
  if [  ${check_bert4keras:8:13} = '0.3.6' ] ; then
    echo "bert4keras installed."
  else
    pip install git+https://www.github.com/bojone/bert4keras.git@v0.3.6
  fi
fi

check_rouge=`pip show rouge | grep "Version"`

if [ ! -n "$check_rouge" ]; then
  pip install rouge
else
  echo "rouge installed."
fi

check_nltk=`pip show nltk | grep "Version"`

if [ ! -n "$check_nltk" ]; then
  pip install nltk
else
  echo "nltk installed."
fi

# check dataset

cd $GLUE_DATA_DIR/$TASK_NAME
if [ ! -f "train.json" ] || [ ! -f "val.json" ] ; then
  echo "Downloading data."
  curl --ftp-skip-pasv-ip ftp://114.115.129.128/CLGE/webqa.zip > webqa.zip
  unzip webqa.zip
  rm webqa.zip
else
  echo "Dataset exists."
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
python ../reading_comprehension.py \
    --dict_path=$ROBERTA_WWM_LARGE_DIR/vocab.txt \
    --config_path=$ROBERTA_WWM_LARGE_DIR/bert_config.json \
    --checkpoint_path=$ROBERTA_WWM_LARGE_DIR/bert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.json \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.json \
    --sample_path=$GLUE_DATA_DIR/$TASK_NAME/sample.json \
    --albert=False \
    --epochs=5 \
    --batch_size=4 \
    --lr=1e-5 \
    --topk=1 \
    --max_p_len=256 \
    --max_q_len=32 \
    --max_a_len=32
