export NCCL_TIMEOUT=60000
set -x

while true
do
    # 生成一个随机端口号，范围在10000到59151之间
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    # 检查端口是否可用
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
torchrun --nnodes 1 --nproc_per_node=8 --master_port $PORT train_CAT_SegFM.py --dist True \
    --num_workers 2 \
    --num_samples 1 \
    --batch_size 1 \
    --log_name SegFM_train_Seg \
    --neg 0 \
    --max_epoch 2000 \
    --store_num 20 \
    --cache_dataset \
    --cache_rate 1