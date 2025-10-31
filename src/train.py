"""
Transformer 训练脚本
包含完整的训练循环、验证、模型保存等功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Transformer, TransformerEncoderOnly
from data_loader import prepare_data, get_data_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 或 "5"，或 "4,5" 如果你想用多卡


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lr_scheduler(optimizer, d_model, warmup_steps=4000):
    """
    Transformer论文中的学习率调度策略
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size * seq_len, vocab_size] (log probabilities)
            target: [batch_size * seq_len]
        """
        assert pred.size(1) == self.vocab_size
        
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        mask = (target != self.padding_idx)
        true_dist = true_dist * mask.unsqueeze(1)
        
        return self.criterion(pred, true_dist) / mask.sum()


def train_epoch(model, dataloader, optimizer, scheduler, criterion, 
                device, clip_grad=1.0, mode='encoder'):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        if mode == 'encoder':
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            output = model(src)  # [batch_size, seq_len, vocab_size]
            
        elif mode == 'seq2seq':
            src, tgt_input, tgt_output = batch
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # 前向传播
            output = model(src, tgt_input)
            tgt = tgt_output
        
        # 计算损失
        output_flat = output.view(-1, output.size(-1))
        tgt_flat = tgt.contiguous().view(-1)
        
        loss = criterion(torch.log_softmax(output_flat, dim=-1), tgt_flat)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        n_tokens = (tgt != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{math.exp(min(loss.item(), 100)):.2f}'
        })
    
    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device, mode='encoder'):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if mode == 'encoder':
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                
            elif mode == 'seq2seq':
                src, tgt_input, tgt_output = batch
                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)
                output = model(src, tgt_input)
                tgt = tgt_output
            
            output_flat = output.view(-1, output.size(-1))
            tgt_flat = tgt.contiguous().view(-1)
            
            loss = criterion(torch.log_softmax(output_flat, dim=-1), tgt_flat)
            
            n_tokens = (tgt != 0).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    return total_loss / total_tokens


def plot_training_curves(train_losses, valid_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Perplexity曲线
    plt.subplot(1, 2, 2)
    train_ppl = [math.exp(min(loss, 100)) for loss in train_losses]
    valid_ppl = [math.exp(min(loss, 100)) for loss in valid_losses]
    plt.plot(train_ppl, label='Train PPL')
    plt.plot(valid_ppl, label='Valid PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


def train(args):
    """主训练函数"""
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("\n准备数据...")
    train_dataset, valid_dataset, test_dataset, vocab = prepare_data(
        data_dir=args.data_dir,
        vocab_path=os.path.join(args.data_dir, 'vocab.pkl'),
        min_freq=args.min_freq,
        max_len=args.max_len,
        mode=args.mode
    )
    
    train_loader, valid_loader, test_loader = get_data_loaders(
        train_dataset, valid_dataset, test_dataset,
        batch_size=args.batch_size,
        mode=args.mode,
        num_workers=args.num_workers
    )
    
    # 创建模型
    print("\n创建模型...")
    if args.mode == 'encoder':
        model = TransformerEncoderOnly(
            vocab_size=vocab.n_words,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_len=args.max_len
        )
    elif args.mode == 'seq2seq':
        model = Transformer(
            src_vocab_size=vocab.n_words,
            tgt_vocab_size=vocab.n_words,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_encoder_layers=args.n_layers,
            n_decoder_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_len=args.max_len
        )
    
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义损失函数
    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(
            vocab_size=vocab.n_words,
            padding_idx=0,
            smoothing=args.label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    # 定义学习率调度器
    if args.use_warmup:
        scheduler = get_lr_scheduler(optimizer, args.d_model, args.warmup_steps)
    else:
        scheduler = None
    
    # 训练循环
    print("\n开始训练...")
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, args.clip_grad, args.mode
        )
        
        # 验证
        valid_loss = evaluate(model, valid_loader, criterion, device, args.mode)
        
        # 计算perplexity
        train_ppl = math.exp(min(train_loss, 100))
        valid_ppl = math.exp(min(valid_loss, 100))
        
        elapsed = time.time() - start_time
        
        print(f'\nEpoch: {epoch:02d} | Time: {elapsed:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}')
        print(f'Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}')
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'vocab': vocab,
                'args': vars(args)
            }
            
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(checkpoint, save_path)
            print(f'最佳模型已保存到 {save_path}')
    
    # 测试
    print("\n在测试集上评估...")
    test_loss = evaluate(model, test_loader, criterion, device, args.mode)
    test_ppl = math.exp(min(test_loss, 100))
    print(f'Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}')
    
    # 保存训练曲线
    os.makedirs('results', exist_ok=True)
    plot_training_curves(
        train_losses, valid_losses,
        os.path.join('results', f'{args.exp_name}_curves.png')
    )
    
    # 保存训练日志
    log = {
        'args': vars(args),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'best_valid_loss': best_valid_loss,
        'model_params': {
            'total': total_params,
            'trainable': trainable_params
        }
    }
    
    log_path = os.path.join('results', f'{args.exp_name}_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"训练日志已保存到 {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Transformer训练脚本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--min_freq', type=int, default=2,
                       help='最小词频')
    parser.add_argument('--max_len', type=int, default=128,
                       help='最大序列长度')
    parser.add_argument('--mode', type=str, default='encoder',
                       choices=['encoder', 'seq2seq'],
                       help='模型模式: encoder或seq2seq')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256,
                       help='模型维度')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='层数')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='标签平滑系数')
    parser.add_argument('--use_warmup', action='store_true',
                       help='是否使用warmup学习率调度')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                       help='Warmup步数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--exp_name', type=str, default='transformer',
                       help='实验名称')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载线程数')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Transformer 训练配置:")
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    train(args)


if __name__ == '__main__':
    main()
