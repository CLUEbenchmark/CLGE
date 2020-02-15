#!/usr/bin/env bash
# @Author: Feng Qiren
# @Date:   2019-2-15

TASK_NAME="lcsts" 
export CUDA_VISIBLE_DEVICES="0"
export GLUE_DATA_DIR=$CURRENT_DIR/../../CLGEdataset

# check python package 

check_keras_layer_normalization=`pip show keras_layer_normalization | grep "Version"`

if [ ! -n "$check_keras_layer_normalization" ]; then
  pip install keras_layer_normalization
else
  echo "keras_layer_normalization installed."
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
  echo "Downloading data."
  curl --ftp-skip-pasv-ip ftp://114.115.129.128/CLGE/lcsts.zip > lcsts.zip
  unzip lcsts.zip
  rm lcsts.zip
else
  echo "Dataset exists."
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
python ../Seq2Seq.py \
    --train_data_path=$GLUE_DATA_DIR/$TASK_NAME/train.tsv \
    --val_data_path=$GLUE_DATA_DIR/$TASK_NAME/val.tsv \
    --sample_path=$GLUE_DATA_DIR/$TASK_NAME/sample.tsv \
    --epochs=10 \
    --batch_size=64 \
    --lr=1e-3 \
    --topk=3 \
    --max_input_len=128 \
    --max_output_len=32
