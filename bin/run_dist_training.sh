#!/bin/bash

# Minimal GPT mock-data run for parallelism testing
# No real dataset, vocab, or merges required

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Get the parallel mode from the command line
parallel_mode=$1

if [ "$parallel_mode" == "tp" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 2
        --pipeline-model-parallel-size 1
    )
elif [ "$parallel_mode" == "pp" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 2
    )
elif [ "$parallel_mode" == "dp" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
    )
else
    echo "Invalid parallel mode: $parallel_mode"
    exit 1
fi

GPUS_PER_NODE=2   # change as needed
MASTER_ADDR=localhost
MASTER_PORT=29500
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
  --num-layers 24
  --hidden-size 1024
  --num-attention-heads 16
  --seq-length 1024
  --max-position-embeddings 1024
  --openai-gelu
  --hidden-dropout 0.1
  --attention-dropout 0.1
  --init-method-std 0.02
  --attention-backend auto
  --transformer-impl local
  --no-persist-layer-norm
  --no-masked-softmax-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 8
    --train-iters 2
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --fp16
    --lr 1.0e-4
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-warmup-fraction .001
    --no-gradient-accumulation-fusion
)

# Mock data + NullTokenizer (no vocab/merge needed)
DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 50257       # GPT-2 default, avoid index OOB
    --merge-file /dev/null   # dummy file path
    --vocab-file /dev/null   # dummy file path
    --seq-length 128
    --split 1,0,0
)

# Disable logging & checkpointing for speed
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100000000
    --save-interval 100000000
    --eval-interval 100000000
    --rerun-mode disabled
    --no-load-optim               # avoid loading optimizer state
    --no-save-optim               # avoid saving optimizer state
    --no-save-rng                 # avoid RNG state save
    # --save /dev/null
    # --load /dev/null
    # --eval-iters 1
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
