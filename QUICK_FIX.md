# 🔧 Seq2Seq Bug 终极修复方案

## 问题
```
RuntimeError: The size of tensor a (127) must match the size of tensor b (8128)
```

## 根本原因
Mask维度不正确。MultiHeadAttention期望：
- 输入mask: `[batch_size, seq_len_q, seq_len_k]`
- 内部会unsqueeze(1)变成: `[batch_size, 1, seq_len_q, seq_len_k]`

## 解决方案

### 修改1: make_src_mask()
找到`make_src_mask`函数（大约在第293行），完整替换为：

```python
def make_src_mask(self, src):
    """创建源序列padding mask
    Returns:
        mask: [batch_size, src_len, src_len]
    """
    # src: [batch_size, src_len]
    src_mask = (src != 0).unsqueeze(1)  # [batch_size, 1, src_len]
    src_mask = src_mask.expand(-1, src.size(1), -1)  # [batch_size, src_len, src_len]
    return src_mask
```

### 修改2: make_tgt_mask()
找到`make_tgt_mask`函数（大约在第303行），完整替换为：

```python
def make_tgt_mask(self, tgt):
    """创建目标序列的look-ahead mask + padding mask
    Returns:
        mask: [batch_size, tgt_len, tgt_len]
    """
    batch_size, tgt_len = tgt.shape
    
    # Padding mask: [batch_size, 1, tgt_len]
    tgt_pad_mask = (tgt != 0).unsqueeze(1)
    
    # Look-ahead mask: [tgt_len, tgt_len]
    tgt_sub_mask = torch.tril(
        torch.ones((tgt_len, tgt_len), device=tgt.device)
    ).bool()
    
    # 组合: PyTorch会自动广播
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    # 结果: [batch_size, tgt_len, tgt_len]
    
    return tgt_mask
```

### 修改3: make_cross_mask()
在`make_tgt_mask`函数之后添加新函数：

```python
def make_cross_mask(self, src, tgt):
    """创建cross-attention的mask
    Args:
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
    Returns:
        mask: [batch_size, tgt_len, src_len]
    """
    src_mask = (src != 0).unsqueeze(1)  # [batch_size, 1, src_len]
    tgt_len = tgt.size(1)
    cross_mask = src_mask.expand(-1, tgt_len, -1)  # [batch_size, tgt_len, src_len]
    return cross_mask
```

### 修改4: forward()函数
找到`forward`函数（大约在第365行），修改为：

```python
def forward(self, src, tgt):
    """
    Args:
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
    Returns:
        output: [batch_size, tgt_len, tgt_vocab_size]
    """
    # 创建mask
    src_mask = self.make_src_mask(src)  # [batch_size, src_len, src_len]
    tgt_mask = self.make_tgt_mask(tgt)  # [batch_size, tgt_len, tgt_len]
    cross_mask = self.make_cross_mask(src, tgt)  # [batch_size, tgt_len, src_len]
    
    # Encoder
    enc_output = self.encode(src, src_mask)
    
    # Decoder (注意：使用cross_mask而不是src_mask)
    dec_output = self.decode(tgt, enc_output, cross_mask, tgt_mask)
    
    # 输出层
    output = self.fc_out(dec_output)
    
    return output
```

### 修改5: decode()函数
找到`decode`函数（大约在第353行），修改参数名：

```python
def decode(self, tgt, enc_output, cross_mask, tgt_mask):
    """Decoder前向传播"""
    # 词嵌入 + 位置编码
    x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
    x = self.pos_encoding(x)
    
    # 通过所有Decoder层
    for layer in self.decoder_layers:
        x = layer(x, enc_output, cross_mask, tgt_mask)
    
    return x
```

### 修改6: DecoderLayer.forward()
找到DecoderLayer的forward函数（大约在第218行），修改参数名：

```python
def forward(self, x, enc_output, cross_mask=None, tgt_mask=None):
    """
    Args:
        x: [batch_size, tgt_len, d_model]
        enc_output: [batch_size, src_len, d_model]
        cross_mask: [batch_size, tgt_len, src_len]
        tgt_mask: [batch_size, tgt_len, tgt_len]
    """
    # Masked多头自注意力 + 残差 + LayerNorm
    self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self_attn_output)
    
    # 编码器-解码器注意力 + 残差 + LayerNorm
    cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
    x = self.norm2(x + cross_attn_output)
    
    # 前馈网络 + 残差 + LayerNorm
    ffn_output = self.ffn(x)
    x = self.norm3(x + ffn_output)
    
    return x
```

## 快速验证

修改后，在项目根目录运行：

```bash
python -c "
import torch
import sys
sys.path.append('src')
from model import Transformer

model = Transformer(1000, 1000, d_model=256, n_heads=4)
src = torch.randint(1, 1000, (4, 10))
tgt = torch.randint(1, 1000, (4, 8))

src_mask = model.make_src_mask(src)
tgt_mask = model.make_tgt_mask(tgt)
cross_mask = model.make_cross_mask(src, tgt)

print('src_mask:', src_mask.shape, '应该是 [4, 10, 10]')
print('tgt_mask:', tgt_mask.shape, '应该是 [4, 8, 8]')
print('cross_mask:', cross_mask.shape, '应该是 [4, 8, 10]')

output = model(src, tgt)
print('output:', output.shape, '应该是 [4, 8, 1000]')
print('✓ 所有测试通过！')
"
```

## 下载修复后的文件

完整的修复后的`model.py`在outputs目录中，可以直接下载替换。

## 重新运行

```bash
bash scripts/run.sh seq2seq
```

应该看到正常的训练输出！

---

**关键点总结：**

| Mask类型 | 正确维度 | 用途 |
|---------|---------|------|
| src_mask | [B, src_len, src_len] | Encoder自注意力 |
| tgt_mask | [B, tgt_len, tgt_len] | Decoder自注意力(masked) |
| cross_mask | [B, tgt_len, src_len] | Decoder交叉注意力 |

**核心问题：** 之前的代码中tgt_pad_mask做了多余的unsqueeze操作，导致维度错误。
