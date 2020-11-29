#!/usr/bin/env bash
# @Author: Li Yudong
# @Date:   2019-12-23

TASK_NAME="csl" 
MODEL_NAME="chinese_L-12_H-768_A-12"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/../../prev_trained_model
export BERT_BASE_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/../../CLGEdataset

# check python package 
check_bert4keras=`pip show bert4keras | grep "Version"`

if [ ! -n "$check_bert4keras" ]; then
  pip install git+https://www.github.com/bojone/bert4keras.git@v0.8.3
else
  if [  ${check_bert4keras:8:13} = '0.8.3' ] ; then
    echo "bert4keras installed."
  else
    pip install git+https://www.github.com/bojone/bert4keras.git@v0.8.3
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
python ../autotitle_baseline.py \
    --dict_path=$BERT_BASE_DIR/vocab.txt \
    --config_path=$BERT_BASE_DIR/bert_config.json \
    --checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt \
    --train_path=$GLUE_DATA_DIR/$TASK_NAME/csl_title_train.json \
    --dev_path=$GLUE_DATA_DIR/$TASK_NAME/csl_title_val.json \
    --test_path=$GLUE_DATA_DIR/$TASK_NAME/csl_title_test.json \
    --test_path=prediction.json \
    --epochs=10 \
    --batch_size=8 \
    --topk=1 \
    --maxlen=256 \
