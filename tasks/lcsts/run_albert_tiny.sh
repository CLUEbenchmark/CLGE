#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-24

TASK_NAME="lcsts"
MODEL_NAME="albert_tiny_zh_google"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export BERT_BASE_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/../../LGECdataset

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

# check dataset

cd $GLUE_DATA_DIR/$TASK_NAME
if [ ! -f "train.tsv" ] || [ ! -f "val.tsv" ] ; then
  echo "Data does not exist."
  exit
fi
echo "Dataset exists."

# download model
if [ ! -d $BERT_PRETRAINED_MODELS_DIR ]; then
  mkdir -p $BERT_PRETRAINED_MODELS_DIR
  echo "makedir $BERT_PRETRAINED_MODELS_DIR"
fi
cd $BERT_PRETRAINED_MODELS_DIR
if [ ! -d $MODEL_NAME ]; then
  mkdir $MODEL_NAME
  cd $MODEL_NAME
  wget https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip
  unzip albert_tiny_zh_google.zip
  rm albert_tiny_zh_google.zip
else
  cd $MODEL_NAME
  if [ ! -f "albert_config_tiny_g.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "albert_model.ckpt.index" ] || [ ! -f "albert_model.ckpt.meta" ] || [ ! -f "albert_model.ckpt.data-00000-of-00001" ]; then
    cd ..
    rm -rf $MODEL_NAME
    mkdir $MODEL_NAME
    cd $MODEL_NAME
    wget https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip
    unzip albert_tiny_zh_google.zip
    rm albert_tiny_zh_google.zip
  else
    echo "model exists"
  fi
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../summary_baseline.py \
    --dict_path=$BERT_BASE_DIR/vocab.txt \
    --config_path=$BERT_BASE_DIR/albert_config_tiny_g.json \
    --checkpoint_path=$BERT_BASE_DIR/albert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.tsv \
    --sample_path=$GLUE_DATA_DIR/$TASK_NAME/sample.tsv \
    --albert=True \
    --epochs=10 \
    --batch_size=16 \
    --lr=1e-5 \
    --topk=1 \
    --max_input_len=128 \
    --max_output_len=24
