#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl"
MODEL_NAME="chinese_wwm_ext_L-12_H-768_A-12"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export BERT_WWM_BASE_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
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

# check dataset

cd $GLUE_DATA_DIR/$TASK_NAME
if [ ! -f "train.tsv" ] || [ ! -f "val.tsv" ] ; then
  echo "Data does not exist."
  exit
fi
echo "Dataset exists."

# download model
if [ ! -d $BERT_WWM_BASE_DIR ]; then
  mkdir -p $BERT_WWM_BASE_DIR
  echo "makedir $BERT_WWM_BASE_DIR"
fi
cd $BERT_WWM_BASE_DIR
if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
  rm *
  wget -c https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip
  unzip chinese_wwm_ext_L-12_H-768_A-12.zip
  rm chinese_wwm_ext_L-12_H-768_A-12.zip
else
  echo "model exists"
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../summary_baseline.py \
    --dict_path=$BERT_WWM_BASE_DIR/vocab.txt \
    --config_path=$BERT_WWM_BASE_DIR/bert_config.json \
    --checkpoint_path=$BERT_WWM_BASE_DIR/bert_model.ckpt \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.tsv \
    --sample_path=$GLUE_DATA_DIR/$TASK_NAME/sample.tsv \
    --albert=False \
    --epochs=10 \
    --batch_size=8 \
    --lr=1e-5 \
    --topk=1 \
    --max_input_len=256 \
    --max_output_len=32
