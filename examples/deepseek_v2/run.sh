#!/bin/bash  

# 参数定义  
ENV="dsw"                          # 运行环境: dlc, dsw  
GPUS_PER_NODE=$1
# MODEL_SIZE="A1.0B"                 # 模型结构参数量级：A21B, A2.4B  
# BATCH_SIZE=1                       # 每卡训练一次迭代样本数: 4, 8  
# GLOBAL_BATCH_SIZE=4               # 全局batch size  
LR="1e-5"                          # 学习率: 1e-5, 5e-5  
MIN_LR="1e-6"                      # 最小学习率: 1e-6, 5e-6  
SEQ_LEN=$2                        # 序列长度  
PAD_LEN=$2                        # Padding长度：100  
PR="bf16"                          # 训练精度: fp16, bf16  
TP=$3                               # 模型并行度  
PP=$4                               # 流水并行度  
EP=$5                               # 专家并行度  
AC="sel"                           # 激活检查点模式: sel, full  
DO=$6                          # 是否使用Megatron版Zero-1降显存优化器: true, false  
FL="true"                          # 是否使用Flash Attention: true, false  
SP=$7                          # 是否使用序列并行: true, false  
SAVE_INTERVAL=10000000                 # 保存ckpt的间隔  
DATASET_PATH="/workspace/megatron_data/deepseek/deepseek-datasets-llamatokenizer/mmap_llama_datasets_text_document"    # 训练数据集路径  
PRETRAIN_CHECKPOINT_PATH=$8  # 预训练模型路径
WARMUP_TOKENS=10000                # 预热token数  
OUTPUT_BASEPATH="/workspace/megatron_data/deepseek/megatron-output"  # 训练输出文件路径  


NUM_LAYERS=$9
HIDDEN_SIZE=${10}
INTERMEDIATE_SIZE=${11}
NUM_ATTN_HEADS=${12}

BATCH_SIZE=${13}                                        # 每卡训练一次迭代样本数: 4, 8  
NUM_MICROBATCHES=${14}
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_MICROBATCHES * GPUS_PER_NODE / PP / TP))      # 全局batch size  
TRAIN_TOKENS=$((GLOBAL_BATCH_SIZE * SEQ_LEN * 10))                 # 训练token数  

# chmod +x ./examples/deepseek_v2/run_pretrain_deepseek.sh  

# 调用运行代码的sh文件并传递参数
bash run_pretrain_deepseek_v2.sh "$GPUS_PER_NODE" "$WORKDIR/Pai-Megatron-Patch" "$BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$LR" "$MIN_LR" "$SEQ_LEN" "$PAD_LEN" "$PR" "$TP" "$PP" "$EP" "$AC" "$DO" "$FL" "$SP" "$SAVE_INTERVAL" "$DATASET_PATH" "$PRETRAIN_CHECKPOINT_PATH" "$TRAIN_TOKENS" "$WARMUP_TOKENS" "$OUTPUT_BASEPATH" "$NUM_LAYERS" "$HIDDEN_SIZE" "$INTERMEDIATE_SIZE" "$NUM_ATTN_HEADS"
