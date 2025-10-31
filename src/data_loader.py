"""
数据加载和预处理模块
支持WikiText-2数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os
import pickle


class Vocabulary:
    """词汇表类"""
    
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = Counter()
        self.n_words = 4
    
    def add_sentence(self, sentence):
        """添加句子到词汇表"""
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def build_from_sentences(self, sentences, min_freq=1):
        """从句子列表构建词汇表"""
        # 统计词频
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] += 1
        
        # 添加高频词
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
    
    def sentence_to_indices(self, sentence, add_eos=False, add_sos=False):
        """将句子转换为索引列表"""
        indices = []
        if add_sos:
            indices.append(self.word2idx['<sos>'])
        
        for word in sentence.split():
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx['<unk>'])
        
        if add_eos:
            indices.append(self.word2idx['<eos>'])
        
        return indices
    
    def indices_to_sentence(self, indices):
        """将索引列表转换为句子"""
        words = []
        for idx in indices:
            if idx in [self.word2idx['<pad>'], self.word2idx['<sos>'], self.word2idx['<eos>']]:
                continue
            words.append(self.idx2word[idx])
        return ' '.join(words)


class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, sentences, vocab, max_len=128, mode='encoder'):
        """
        Args:
            sentences: 句子列表
            vocab: 词汇表对象
            max_len: 最大序列长度
            mode: 'encoder' 或 'seq2seq'
        """
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len
        self.mode = mode
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        if self.mode == 'encoder':
            # Encoder-only模式: 用于语言建模
            indices = self.vocab.sentence_to_indices(sentence, add_eos=True)
            
            # 截断
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]
            
            # 输入是去掉最后一个token
            # 目标是去掉第一个token (预测下一个词)
            src = torch.tensor(indices[:-1], dtype=torch.long)
            tgt = torch.tensor(indices[1:], dtype=torch.long)
            
            return src, tgt
        
        elif self.mode == 'seq2seq':
            # Seq2seq模式 (这里简化为copy任务用于演示)
            src_indices = self.vocab.sentence_to_indices(sentence, add_eos=True)
            tgt_indices = self.vocab.sentence_to_indices(sentence, add_sos=True, add_eos=True)
            
            if len(src_indices) > self.max_len:
                src_indices = src_indices[:self.max_len]
            if len(tgt_indices) > self.max_len:
                tgt_indices = tgt_indices[:self.max_len]
            
            src = torch.tensor(src_indices, dtype=torch.long)
            tgt = torch.tensor(tgt_indices, dtype=torch.long)
            
            return src, tgt[:-1], tgt[1:]  # src, tgt_input, tgt_output


def collate_fn_encoder(batch):
    """Encoder模式的batch处理函数"""
    srcs, tgts = zip(*batch)
    
    # Padding
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
    
    return srcs_padded, tgts_padded


def collate_fn_seq2seq(batch):
    """Seq2seq模式的batch处理函数"""
    srcs, tgt_inputs, tgt_outputs = zip(*batch)
    
    # Padding
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_inputs_padded = pad_sequence(tgt_inputs, batch_first=True, padding_value=0)
    tgt_outputs_padded = pad_sequence(tgt_outputs, batch_first=True, padding_value=0)
    
    return srcs_padded, tgt_inputs_padded, tgt_outputs_padded


def load_wikitext2(data_dir='data'):
    """加载WikiText-2数据集"""
    train_file = os.path.join(data_dir, 'wiki.train.tokens')
    valid_file = os.path.join(data_dir, 'wiki.valid.tokens')
    test_file = os.path.join(data_dir, 'wiki.test.tokens')
    
    def read_file(file_path):
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和标题行
                if line and not line.startswith('='):
                    sentences.append(line.lower())
        return sentences
    
    train_sentences = read_file(train_file)
    valid_sentences = read_file(valid_file)
    test_sentences = read_file(test_file)
    
    return train_sentences, valid_sentences, test_sentences


def prepare_data(data_dir='data', vocab_path='data/vocab.pkl', 
                 min_freq=2, max_len=128, mode='encoder'):
    """准备数据和词汇表"""
    
    # 加载数据
    print("正在加载WikiText-2数据集...")
    train_sentences, valid_sentences, test_sentences = load_wikitext2(data_dir)
    
    print(f"训练集: {len(train_sentences)} 句")
    print(f"验证集: {len(valid_sentences)} 句")
    print(f"测试集: {len(test_sentences)} 句")
    
    # 构建或加载词汇表
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载词汇表...")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        print("构建词汇表...")
        vocab = Vocabulary()
        vocab.build_from_sentences(train_sentences, min_freq=min_freq)
        
        # 保存词汇表
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"词汇表已保存到 {vocab_path}")
    
    print(f"词汇表大小: {vocab.n_words}")
    
    # 创建数据集
    train_dataset = TextDataset(train_sentences, vocab, max_len, mode)
    valid_dataset = TextDataset(valid_sentences, vocab, max_len, mode)
    test_dataset = TextDataset(test_sentences, vocab, max_len, mode)
    
    return train_dataset, valid_dataset, test_dataset, vocab


def get_data_loaders(train_dataset, valid_dataset, test_dataset, 
                    batch_size=32, mode='encoder', num_workers=0):
    """创建数据加载器"""
    
    collate_fn = collate_fn_encoder if mode == 'encoder' else collate_fn_seq2seq
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    train_dataset, valid_dataset, test_dataset, vocab = prepare_data(
        data_dir='../data',
        vocab_path='../data/vocab.pkl'
    )
    
    train_loader, valid_loader, test_loader = get_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=4
    )
    
    # 查看一个batch
    for src, tgt in train_loader:
        print(f"Source shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Source example: {vocab.indices_to_sentence(src[0].tolist())}")
        print(f"Target example: {vocab.indices_to_sentence(tgt[0].tolist())}")
        break
