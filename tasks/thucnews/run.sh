# run task

export DATA_DIR="../../CLGEdataset/thucnews/data"
export CUDA_VISIBLE_DEVICES="0"

echo "Start running..."
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python ../title_generation.py \
    --train_data_path=$DATA_DIR \
    --seq2seq_config_path="./" \
    --min_count=32 \
    --maxlen=400 \
    --char_size=128 \
    --topk=3 \
    --z_dim=128 \
    --epochs=100 \
    --batch_size=64
