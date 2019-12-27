#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="lcsts" 
MODEL_NAME="chinese_wwm_ext_L-12_H-768_A-12"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="1"
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
else
  echo "Dataset exists."
fi

# download model
if [ ! -d $BERT_PRETRAINED_MODELS_DIR ]; then
  mkdir -p $BERT_PRETRAINED_MODELS_DIR
  echo "makedir $BERT_PRETRAINED_MODELS_DIR"
fi
cd $BERT_PRETRAINED_MODELS_DIR
if [ ! -d $MODEL_NAME ]; then
  wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
  unzip chinese_L-12_H-768_A-12.zip
  rm chinese_L-12_H-768_A-12.zip
else
  cd $MODEL_NAME
  if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
    cd ..
    rm -rf $MODEL_NAME
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip chinese_L-12_H-768_A-12.zip
    rm chinese_L-12_H-768_A-12.zip
  else
    echo "model exists"
  fi
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../summary_baseline.py \
    --dict_path=$BERT_BASE_DIR/vocab.txt \
    --config_path=$BERT_BASE_DIR/bert_config.json \
    --checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.tsv \
    --sample_path=$GLUE_DATA_DIR/$TASK_NAME/sample.tsv \
    --albert=False \
    --epochs=30 \
    --batch_size=16 \
    --lr=1e-5 \
    --topk=1 \
    --max_input_len=128 \
    --max_output_len=32
