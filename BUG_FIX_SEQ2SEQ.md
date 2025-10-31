# 🐛 Seq2Seq模式Bug修复说明

## 问题描述

运行 `bash scripts/run.sh seq2seq` 时出现错误：
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (8192) at non-singleton dimension 1
```

## 问题原因

在原始实现中，mask的维度处理有问题：

1. **MultiHeadAttention期望的mask格式**：
   - 输入: `[batch_size, seq_len_q, seq_len_k]`
   - 内部会unsqueeze(1)变成: `[batch_size, 1, seq_len_q, seq_len_k]`

2. **原始代码的问题**：
   - `make_src_mask()`返回: `[batch_size, 1, 1, src_len]`
   - `make_tgt_mask()`返回: `[batch_size, 1, tgt_len, tgt_len]`
   - 这些mask在MultiHeadAttention中又被unsqueeze(1)，导致维度不匹配

3. **Cross-attention的特殊性**：
   - Query来自decoder (长度tgt_len)
   - Key/Value来自encoder (长度src_len)
   - 需要专门的cross_mask: `[batch_size, tgt_len, src_len]`

## 解决方案

### 修改1: make_src_mask()

**修改前**:
```python
def make_src_mask(self, src):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    # 返回 [batch_size, 1, 1, src_len]
    return src_mask
```

**修改后**:
```python
def make_src_mask(self, src):
    """创建源序列padding mask
    Returns:
        mask: [batch_size, 1, src_len] - 适配MultiHeadAttention的输入格式
    """
    src_mask = (src != 0).unsqueeze(1)  # [batch_size, 1, src_len]
    return src_mask
```

### 修改2: make_tgt_mask()

**修改前**:
```python
def make_tgt_mask(self, tgt):
    batch_size, tgt_len = tgt.shape
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask
```

**修改后**:
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
    
    # 组合: [batch_size, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask.unsqueeze(1) & tgt_sub_mask.unsqueeze(0)
    
    return tgt_mask
```

### 修改3: 新增make_cross_mask()

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
    cross_mask = src_mask.expand(-1, tgt_len, -1)
    return cross_mask
```

### 修改4: forward()函数

**修改前**:
```python
def forward(self, src, tgt):
    src_mask = self.make_src_mask(src)
    tgt_mask = self.make_tgt_mask(tgt)
    
    enc_output = self.encode(src, src_mask)
    dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)  # ❌ 错误
    
    output = self.fc_out(dec_output)
    return output
```

**修改后**:
```python
def forward(self, src, tgt):
    src_mask = self.make_src_mask(src)  # [batch_size, 1, src_len]
    tgt_mask = self.make_tgt_mask(tgt)  # [batch_size, tgt_len, tgt_len]
    cross_mask = self.make_cross_mask(src, tgt)  # [batch_size, tgt_len, src_len]
    
    enc_output = self.encode(src, src_mask)
    dec_output = self.decode(tgt, enc_output, cross_mask, tgt_mask)  # ✅ 正确
    
    output = self.fc_out(dec_output)
    return output
```

### 修改5: decode()和DecoderLayer

```python
# decode函数
def decode(self, tgt, enc_output, cross_mask, tgt_mask):  # 参数名改为cross_mask
    x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
    x = self.pos_encoding(x)
    
    for layer in self.decoder_layers:
        x = layer(x, enc_output, cross_mask, tgt_mask)  # 使用cross_mask
    
    return x

# DecoderLayer.forward
def forward(self, x, enc_output, cross_mask=None, tgt_mask=None):
    # Self-attention
    self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self_attn_output)
    
    # Cross-attention (使用cross_mask)
    cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
    x = self.norm2(x + cross_attn_output)
    
    # FFN
    ffn_output = self.ffn(x)
    x = self.norm3(x + ffn_output)
    
    return x
```

## 如何应用修复

### 方法1: 手动修改代码

1. 打开 `src/model.py`
2. 找到 `Transformer` 类
3. 按照上述说明修改5个部分

### 方法2: 下载修复后的文件

修复后的完整文件已保存在outputs目录：
- 路径: `transformer_project/src/model.py`
- 直接复制到你的项目中替换原文件

### 方法3: 使用备份恢复（如果需要）

如果修改出错，可以从备份恢复：
```bash
cp src/model.py.backup src/model.py
```

## 验证修复

修复后，重新运行：
```bash
bash scripts/run.sh seq2seq
```

应该看到：
```
使用设备: cuda
准备数据...
创建模型...
模型总参数: 29,603,052
可训练参数: 29,603,052
开始训练...
Training: 100%|████████████| 275/275 [00:XX<00:00, X.XXit/s, loss=6.XXXX, ppl=XXX.XX]
Epoch: 01 | Time: XXXs
Train Loss: 6.XXXX | Train PPL: XXX.XX
Valid Loss: 6.XXXX | Valid PPL: XXX.XX
```

## Mask维度总结

| Mask类型 | 维度 | 用途 |
|---------|------|------|
| src_mask | [batch, 1, src_len] | Encoder self-attention |
| tgt_mask | [batch, tgt_len, tgt_len] | Decoder self-attention (masked) |
| cross_mask | [batch, tgt_len, src_len] | Decoder cross-attention |

**关键点**：
- MultiHeadAttention期望输入mask为3D: `[batch, seq_q, seq_k]`
- 内部会自动unsqueeze(1)变成4D: `[batch, 1, seq_q, seq_k]`
- 不要在make_mask函数中预先做这个unsqueeze！

## 其他注意事项

1. **Encoder-only模式不受影响**：
   - `TransformerEncoderOnly`类使用简单的self-attention
   - mask处理更简单，不需要cross_mask

2. **如果遇到其他维度错误**：
   - 检查batch_size是否正确
   - 检查序列长度是否在max_len范围内
   - 使用`print(tensor.shape)`调试

3. **性能优化**：
   - 修复后的代码不会影响性能
   - 训练速度应该与之前相同

## 测试建议

修复后，建议进行快速测试：
```bash
# 快速测试（3 epochs）
python src/train.py \
  --mode seq2seq \
  --epochs 3 \
  --batch_size 32 \
  --exp_name test_seq2seq

# 如果成功，再运行完整训练
bash scripts/run.sh seq2seq
```

---

**修复完成！** 现在seq2seq模式应该可以正常运行了。

如有其他问题，请检查：
1. PyTorch版本 (建议 >= 2.0.0)
2. CUDA是否可用
3. 数据集是否正确加载
