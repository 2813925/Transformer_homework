# Transformer 从零实现

基于PyTorch的Transformer完整实现，包含Encoder-Decoder架构、训练流程和消融实验。

##  项目概述

本项目从零实现了完整的Transformer模型，用于"大模型基础与应用"课程期中作业。主要特性：

- ✅ **完整架构**: 实现Encoder和Decoder，支持Encoder-only和完整Seq2Seq模式
- ✅ **核心组件**: Multi-Head Attention、Position-wise FFN、Residual+LayerNorm、Positional Encoding
- ✅ **训练稳定性**: AdamW优化器、学习率warmup、梯度裁剪、标签平滑
- ✅ **消融实验**: 系统性测试各组件对性能的影响
- ✅ **完整文档**: 详细的数学推导和实现说明

##  项目结构

```
transformer_project/
├── src/
│   ├── model.py           # Transformer模型实现
│   ├── data_loader.py     # 数据加载和预处理
│   ├── train.py           # 训练脚本
│   └── ablation.py        # 消融实验脚本
├── scripts/
│   ├── run.sh             # 训练运行脚本
│   └── download_data.py   # 数据下载脚本
├── configs/               # 配置文件目录
├── results/               # 训练结果、曲线图
├── checkpoints/           # 模型检查点
├── data/                  # 数据集目录
├── requirements.txt       # 依赖包
└── README.md             # 本文件
```

##  数据集

**使用数据集**: WikiText-2

- **描述**: 约2M tokens的语言建模数据集，来自Wikipedia优质文章
- **任务类型**: 语言建模 (Language Modeling)
- **数据规模**: 
  - 训练集: ~600K tokens
  - 验证集: ~60K tokens  
  - 测试集: ~60K tokens
- **下载链接**: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/



### 1. 环境配置

```bash
# 创建conda环境
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖
pip install torch numpy matplotlib tqdm pandas
# 或
pip install -r requirements.txt
```

### 2. 下载数据

**方法1: 使用脚本自动下载**
```bash
python scripts/download_data.py
```

**方法2: 手动下载**
1. 访问 https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
2. 下载并解压到 `data/` 目录
3. 确保目录结构为:
```
data/
├── wiki.train.tokens
├── wiki.valid.tokens
└── wiki.test.tokens
```

### 3. 训练模型

**训练Encoder-only模型 (语言建模)**
```bash
bash scripts/run.sh encoder
```

**训练完整Encoder-Decoder模型**
```bash
bash scripts/run.sh seq2seq
```

**运行消融实验**
```bash
bash scripts/run.sh ablation
```

### 4. 自定义训练

```bash
python src/train.py \
  --data_dir data \
  --mode encoder \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --d_ff 1024 \
  --batch_size 64 \
  --epochs 20 \
  --lr 0.0001 \
  --seed 42 \
  --exp_name my_experiment
```

##  模型架构

### 核心组件

#### 1. Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

#### 2. Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### 3. Position-wise Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

#### 4. Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 256 | 模型维度 |
| n_heads | 4 | 注意力头数 |
| n_layers | 4 | Encoder/Decoder层数 |
| d_ff | 1024 | 前馈网络维度 |
| dropout | 0.1 | Dropout概率 |
| max_len | 128 | 最大序列长度 |

**参数量统计** (Encoder-only, 默认配置):
- 总参数: ~8.5M
- 可训练参数: ~8.5M

##  实验设置

### 训练超参数

| 超参数 | 值 |
|--------|-----|
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.98, ε=1e-9) |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| Label Smoothing | 0.1 |
| Epochs | 20 |
| 随机种子 | 42 |

### 消融实验

测试以下变量对性能的影响：

1. **位置编码**: 移除位置编码
2. **注意力头数**: 2头 vs 4头 vs 8头
3. **模型层数**: 2层 vs 4层 vs 6层
4. **Dropout**: 0.0 vs 0.1 vs 0.3
5. **模型规模**: d_model=128 vs 256
6. **标签平滑**: 有 vs 无

##  结果示例

训练完成后，`results/` 目录将包含：

- `transformer_encoder_curves.png`: 训练/验证损失曲线
- `transformer_encoder_log.json`: 详细训练日志
- `ablation_results.csv`: 消融实验结果表格
- `ablation_comparison.png`: 消融实验对比图

**预期结果** (WikiText-2, Encoder-only):
- Test Loss: ~4.5-5.0
- Test Perplexity: ~90-150

##  硬件要求

**最低配置**:
- CPU: 4核
- 内存: 8GB
- 存储: 2GB

**推荐配置**:
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA支持)
- CPU: 8核
- 内存: 16GB
- 存储: 5GB

**训练时间估计** (默认配置):
- CPU: ~4-6小时/20 epochs
- GPU (RTX 3060): ~30-45分钟/20 epochs

##  完整命令行示例

```bash
# 1. 设置环境
conda create -n transformer python=3.10
conda activate transformer
pip install -r requirements.txt

# 2. 下载数据
python scripts/download_data.py

# 3. 训练Encoder模型 (可复现)
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

# 4. 训练Encoder-Decoder模型
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
  --seed 42 \
  --exp_name transformer_seq2seq

# 5. 运行消融实验
python src/ablation.py \
  --data_dir data \
  --mode encoder \
  --epochs 15 \
  --seed 42
```

##  代码关键实现

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        # 线性变换并分割成多头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和并拼接
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        return self.W_O(output)
```

##  常见问题

**Q: 下载数据集失败？**
A: 可以手动从 https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip 下载

**Q: CUDA out of memory？**
A: 减小batch_size (如改为32或16)，或减小模型规模 (d_model=128)

**Q: 训练速度太慢？**
A: 确保使用GPU训练。检查 `torch.cuda.is_available()` 是否返回True

**Q: Loss不下降？**
A: 检查学习率是否合适，可以尝试降低到5e-5；确保数据预处理正确

##  参考文献

[1] Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

[2] Merity, S., et al. (2016). Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*.

##  信息


- **课程**: 大模型基础与应用
- **学期**: 2025年秋季

##  许可证

MIT License

##  致谢

感谢Salesforce Research提供WikiText-2数据集，感谢PyTorch团队提供优秀的深度学习框架。

---

**最后更新**: 2025年10月
