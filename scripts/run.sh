#!/bin/bash

# Transformer训练脚本
# 用法: bash scripts/run.sh [encoder|seq2seq|ablation]

MODE=${1:-encoder}

echo "=========================================="
echo "Transformer 训练脚本"
echo "模式: $MODE"
echo "=========================================="

case $MODE in
  encoder)
    echo "训练 Encoder-only Transformer..."
    python src/train.py \
      --data_dir data \
      --mode encoder \
      --d_model 256 \
      --n_heads 4 \
      --n_layers 4 \
      --d_ff 1024 \
      --dropout 0.1 \
      --batch_size 64 \
      --epochs 20 \
      --lr 0.0001 \
      --weight_decay 0.01 \
      --clip_grad 1.0 \
      --label_smoothing 0.1 \
      --seed 42 \
      --exp_name transformer_encoder
    ;;
    
  seq2seq)
    echo "训练 Encoder-Decoder Transformer..."
    python src/train.py \
      --data_dir data \
      --mode seq2seq \
      --d_model 256 \
      --n_heads 4 \
      --n_layers 4 \
      --d_ff 1024 \
      --dropout 0.1 \
      --batch_size 64 \
      --epochs 20 \
      --lr 0.0001 \
      --weight_decay 0.01 \
      --clip_grad 1.0 \
      --label_smoothing 0.1 \
      --seed 42 \
      --exp_name transformer_seq2seq
    ;;
    
  ablation)
    echo "运行消融实验..."
    python src/ablation.py \
      --data_dir data \
      --mode encoder \
      --d_model 256 \
      --n_heads 4 \
      --n_layers 4 \
      --d_ff 1024 \
      --dropout 0.1 \
      --batch_size 64 \
      --epochs 15 \
      --lr 0.0001 \
      --seed 42
    ;;
    
  *)
    echo "未知模式: $MODE"
    echo "用法: bash scripts/run.sh [encoder|seq2seq|ablation]"
    exit 1
    ;;
esac

echo "=========================================="
echo "训练完成！"
echo "=========================================="
