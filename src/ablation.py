"""
消融实验脚本 (机器翻译任务)
测试不同组件对模型性能的影响
"""
import torch
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from train import train, set_seed
import sys


def run_ablation_study(base_args):
    """运行消融实验"""
    
    results = []
    
    # 实验配置列表
    experiments = [
        {
            'name': 'baseline',
            'description': '完整模型 (基线)',
            'modifications': {}
        },
        {
            'name': 'fewer_heads',
            'description': '减少注意力头数 (2个头)',
            'modifications': {'n_heads': 2}
        },
        {
            'name': 'more_heads',
            'description': '增加注意力头数 (8个头)',
            'modifications': {'n_heads': 8}
        },
        {
            'name': 'fewer_layers',
            'description': '减少层数 (2层)',
            'modifications': {'n_layers': 2}
        },
        {
            'name': 'more_layers',
            'description': '增加层数 (6层)',
            'modifications': {'n_layers': 6}
        },
        {
            'name': 'no_dropout',
            'description': '移除Dropout',
            'modifications': {'dropout': 0.0}
        },
        {
            'name': 'higher_dropout',
            'description': '更高Dropout (0.3)',
            'modifications': {'dropout': 0.3}
        },
        {
            'name': 'smaller_model',
            'description': '更小的模型维度 (128)',
            'modifications': {'d_model': 128, 'd_ff': 512}
        },
        {
            'name': 'no_label_smoothing',
            'description': '移除标签平滑',
            'modifications': {'label_smoothing': 0.0}
        }
    ]
    
    print("=" * 80)
    print("开始消融实验 (机器翻译任务)")
    print("=" * 80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n实验 {i}/{len(experiments)}: {exp['description']}")
        print("-" * 80)
        
        # 创建实验参数
        exp_args = argparse.Namespace(**vars(base_args))
        
        # 应用修改
        for key, value in exp['modifications'].items():
            setattr(exp_args, key, value)
        
        # 设置实验名称
        exp_args.exp_name = f"ablation_{exp['name']}"
        
        # 运行训练
        try:
            train(exp_args)
            
            # 读取结果
            log_path = os.path.join('results', f"{exp_args.exp_name}_log.json")
            with open(log_path, 'r') as f:
                log = json.load(f)
            
            results.append({
                'experiment': exp['name'],
                'description': exp['description'],
                'test_loss': log['test_loss'],
                'test_ppl': log['test_ppl'],
                'best_valid_loss': log['best_valid_loss'],
                'model_params': log['model_params']['trainable']
            })
            
            print(f"✓ 实验完成: Test Loss = {log['test_loss']:.4f}, Test PPL = {log['test_ppl']:.2f}")
            
        except Exception as e:
            print(f"✗ 实验失败: {str(e)}")
            results.append({
                'experiment': exp['name'],
                'description': exp['description'],
                'test_loss': None,
                'test_ppl': None,
                'best_valid_loss': None,
                'model_params': None,
                'error': str(e)
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', 'ablation_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n消融实验结果已保存到 {results_path}")
    
    # 可视化结果
    plot_ablation_results(results_df)
    
    return results_df


def plot_ablation_results(results_df):
    """可视化消融实验结果"""
    
    # 过滤掉失败的实验
    valid_results = results_df[results_df['test_loss'].notna()].copy()
    
    if len(valid_results) == 0:
        print("没有有效的实验结果可以可视化")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Test Loss对比
    ax1 = axes[0]
    bars1 = ax1.barh(valid_results['experiment'], valid_results['test_loss'])
    ax1.set_xlabel('Test Loss', fontsize=12)
    ax1.set_title('消融实验: Test Loss对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 标注baseline
    baseline_idx = valid_results[valid_results['experiment'] == 'baseline'].index
    if len(baseline_idx) > 0:
        bars1[valid_results.index.get_loc(baseline_idx[0])].set_color('red')
    
    # Test Perplexity对比
    ax2 = axes[1]
    bars2 = ax2.barh(valid_results['experiment'], valid_results['test_ppl'])
    ax2.set_xlabel('Test Perplexity', fontsize=12)
    ax2.set_title('消融实验: Test Perplexity对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 标注baseline
    if len(baseline_idx) > 0:
        bars2[valid_results.index.get_loc(baseline_idx[0])].set_color('red')
    
    plt.tight_layout()
    save_path = os.path.join('results', 'ablation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"消融实验对比图已保存到 {save_path}")
    
    # 打印详细结果表格
    print("\n" + "=" * 80)
    print("消融实验详细结果:")
    print("=" * 80)
    print(valid_results[['experiment', 'description', 'test_loss', 'test_ppl', 'model_params']].to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Transformer消融实验 (机器翻译)')
    
    # 基础配置
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--min_freq', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=128)
    
    # 模型参数 (基线配置)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15,
                       help='每个消融实验的训练轮数 (减少以加快实验)')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_warmup', action='store_true')
    parser.add_argument('--warmup_steps', type=int, default=4000)
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # 运行消融实验
    results = run_ablation_study(args)


if __name__ == '__main__':
    main()
