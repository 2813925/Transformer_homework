"""
Transformer 模型实现
实现了完整的 Encoder-Decoder 架构
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query矩阵 [batch_size, n_heads, seq_len, d_k]
            K: Key矩阵 [batch_size, n_heads, seq_len, d_k]
            V: Value矩阵 [batch_size, n_heads, seq_len, d_v]
            mask: 掩码 [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len, d_v]
            attn: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码 (将padding位置或未来位置设为极小值)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 注意力加权求和
        output = torch.matmul(attn, V)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, seq_len_q, d_model]
            K: [batch_size, seq_len_k, d_model]
            V: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k]
        """
        batch_size = Q.size(0)
        
        # 线性变换并分割成多头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 调整mask维度以适配多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
        
        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终线性变换
        output = self.W_O(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """位置前馈神经网络 (Position-wise FFN)"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        output = self.fc1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    """位置编码 (Sinusoidal Positional Encoding)"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Transformer Encoder层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈神经网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接 + Layer Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈神经网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, cross_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_len, d_model]
            enc_output: [batch_size, src_len, d_model]
            cross_mask: [batch_size, tgt_len, src_len] (padding mask for encoder output)
            tgt_mask: [batch_size, tgt_len, tgt_len] (look-ahead mask for decoder)
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


class Transformer(nn.Module):
    """完整的Transformer模型 (Encoder-Decoder架构)"""
    
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """创建源序列padding mask
        Returns:
            mask: [batch_size, src_len, src_len] - 用于encoder self-attention
        """
        # src: [batch_size, src_len]
        # 创建padding mask: [batch_size, 1, src_len]
        src_mask = (src != 0).unsqueeze(1)  # [batch_size, 1, src_len]
        # 扩展到方阵: [batch_size, src_len, src_len]
        src_mask = src_mask.expand(-1, src.size(1), -1)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """创建目标序列的look-ahead mask + padding mask
        Returns:
            mask: [batch_size, tgt_len, tgt_len] - 适配MultiHeadAttention的输入格式
        """
        batch_size, tgt_len = tgt.shape
        
        # Padding mask: [batch_size, tgt_len] -> [batch_size, 1, tgt_len]
        tgt_pad_mask = (tgt != 0).unsqueeze(1)  # [batch_size, 1, tgt_len]
        
        # Look-ahead mask (下三角矩阵): [tgt_len, tgt_len]
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()  # [tgt_len, tgt_len]
        
        # 组合两种mask
        # tgt_pad_mask: [batch_size, 1, tgt_len] 会广播到 [batch_size, tgt_len, tgt_len]
        # tgt_sub_mask: [tgt_len, tgt_len] 会广播到 [batch_size, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # 结果: [batch_size, tgt_len, tgt_len]
        
        return tgt_mask
    
    def make_cross_mask(self, src, tgt):
        """创建cross-attention的mask (用于decoder关注encoder输出)
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            mask: [batch_size, tgt_len, src_len] - decoder attend to encoder
        """
        # 只需要src的padding mask
        # [batch_size, 1, src_len] -> [batch_size, tgt_len, src_len]
        src_mask = (src != 0).unsqueeze(1)  # [batch_size, 1, src_len]
        tgt_len = tgt.size(1)
        cross_mask = src_mask.expand(-1, tgt_len, -1)  # [batch_size, tgt_len, src_len]
        return cross_mask
    
    def encode(self, src, src_mask):
        """Encoder前向传播"""
        # 词嵌入 + 位置编码
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有Encoder层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, enc_output, cross_mask, tgt_mask):
        """Decoder前向传播"""
        # 词嵌入 + 位置编码
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有Decoder层
        for layer in self.decoder_layers:
            x = layer(x, enc_output, cross_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建mask
        src_mask = self.make_src_mask(src)  # [batch_size, 1, src_len] for encoder self-attn
        tgt_mask = self.make_tgt_mask(tgt)  # [batch_size, tgt_len, tgt_len] for decoder self-attn
        cross_mask = self.make_cross_mask(src, tgt)  # [batch_size, tgt_len, src_len] for cross-attn
        
        # Encoder
        enc_output = self.encode(src, src_mask)
        
        # Decoder
        dec_output = self.decode(tgt, enc_output, cross_mask, tgt_mask)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output


class TransformerEncoderOnly(nn.Module):
    """仅Encoder的Transformer (用于语言建模)"""
    
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len]
            mask: [batch_size, seq_len, seq_len]
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过所有Encoder层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 输出
        output = self.fc_out(x)
        
        return output
