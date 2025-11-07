"""
数据加载和预处理模块
支持IWSLT2017 EN-DE机器翻译数据集
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


class TranslationDataset(Dataset):
    """机器翻译数据集类 (Encoder-Decoder)"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=128):
        """
        Args:
            src_sentences: 源语言句子列表 (English)
            tgt_sentences: 目标语言句子列表 (German)
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_len: 最大序列长度
        """
        assert len(src_sentences) == len(tgt_sentences), "源语言和目标语言句子数量必须相同"
        
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # 源序列: 只加<eos>
        src_indices = self.src_vocab.sentence_to_indices(src_sentence, add_eos=True)
        
        # 目标序列: 加<sos>和<eos>
        tgt_indices = self.tgt_vocab.sentence_to_indices(tgt_sentence, add_sos=True, add_eos=True)
        
        # 截断
        if len(src_indices) > self.max_len:
            src_indices = src_indices[:self.max_len]
        if len(tgt_indices) > self.max_len:
            tgt_indices = tgt_indices[:self.max_len]
        
        src = torch.tensor(src_indices, dtype=torch.long)
        tgt = torch.tensor(tgt_indices, dtype=torch.long)
        
        # 返回: src, tgt_input (<sos>...),  tgt_output (...<eos>)
        return src, tgt[:-1], tgt[1:]


def collate_fn_seq2seq(batch):
    """Seq2seq模式的batch处理函数"""
    srcs, tgt_inputs, tgt_outputs = zip(*batch)
    
    # Padding
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_inputs_padded = pad_sequence(tgt_inputs, batch_first=True, padding_value=0)
    tgt_outputs_padded = pad_sequence(tgt_outputs, batch_first=True, padding_value=0)
    
    return srcs_padded, tgt_inputs_padded, tgt_outputs_padded


def load_iwslt2017(data_dir='data'):
    """加载IWSLT2017 EN-DE数据集"""
    
    def read_file(file_path):
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    sentences.append(line.lower())
        return sentences
    
    # 读取源语言 (English)
    train_src = read_file(os.path.join(data_dir, 'train.en'))
    valid_src = read_file(os.path.join(data_dir, 'valid.en'))
    test_src = read_file(os.path.join(data_dir, 'test.en'))
    
    # 读取目标语言 (German)
    train_tgt = read_file(os.path.join(data_dir, 'train.de'))
    valid_tgt = read_file(os.path.join(data_dir, 'valid.de'))
    test_tgt = read_file(os.path.join(data_dir, 'test.de'))
    
    return (train_src, train_tgt), (valid_src, valid_tgt), (test_src, test_tgt)


def prepare_data(data_dir='data', 
                 src_vocab_path='data/vocab_en.pkl',
                 tgt_vocab_path='data/vocab_de.pkl',
                 min_freq=2, max_len=128):
    """准备机器翻译数据和词汇表"""
    
    # 加载数据
    print("正在加载IWSLT2017 EN-DE数据集...")
    (train_src, train_tgt), (valid_src, valid_tgt), (test_src, test_tgt) = load_iwslt2017(data_dir)
    
    print(f"训练集: {len(train_src)} 句对")
    print(f"验证集: {len(valid_src)} 句对")
    print(f"测试集: {len(test_src)} 句对")
    
    # 构建或加载源语言词汇表 (English)
    if os.path.exists(src_vocab_path):
        print(f"从 {src_vocab_path} 加载源语言词汇表...")
        with open(src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
    else:
        print("构建源语言词汇表 (English)...")
        src_vocab = Vocabulary()
        src_vocab.build_from_sentences(train_src, min_freq=min_freq)
        
        os.makedirs(os.path.dirname(src_vocab_path), exist_ok=True)
        with open(src_vocab_path, 'wb') as f:
            pickle.dump(src_vocab, f)
        print(f"源语言词汇表已保存到 {src_vocab_path}")
    
    print(f"源语言词汇表大小: {src_vocab.n_words}")
    
    # 构建或加载目标语言词汇表 (German)
    if os.path.exists(tgt_vocab_path):
        print(f"从 {tgt_vocab_path} 加载目标语言词汇表...")
        with open(tgt_vocab_path, 'rb') as f:
            tgt_vocab = pickle.load(f)
    else:
        print("构建目标语言词汇表 (German)...")
        tgt_vocab = Vocabulary()
        tgt_vocab.build_from_sentences(train_tgt, min_freq=min_freq)
        
        os.makedirs(os.path.dirname(tgt_vocab_path), exist_ok=True)
        with open(tgt_vocab_path, 'wb') as f:
            pickle.dump(tgt_vocab, f)
        print(f"目标语言词汇表已保存到 {tgt_vocab_path}")
    
    print(f"目标语言词汇表大小: {tgt_vocab.n_words}")
    
    # 创建数据集
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    valid_dataset = TranslationDataset(valid_src, valid_tgt, src_vocab, tgt_vocab, max_len)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)
    
    return train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab


def get_data_loaders(train_dataset, valid_dataset, test_dataset, 
                    batch_size=32, num_workers=0):
    """创建数据加载器"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_seq2seq,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_seq2seq,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_seq2seq,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab = prepare_data(
        data_dir='../data',
        src_vocab_path='../data/vocab_en.pkl',
        tgt_vocab_path='../data/vocab_de.pkl'
    )
    
    train_loader, valid_loader, test_loader = get_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=4
    )
    
    # 查看一个batch
    for src, tgt_in, tgt_out in train_loader:
        print(f"Source shape: {src.shape}")
        print(f"Target input shape: {tgt_in.shape}")
        print(f"Target output shape: {tgt_out.shape}")
        print(f"Source example (EN): {src_vocab.indices_to_sentence(src[0].tolist())}")
        print(f"Target example (DE): {tgt_vocab.indices_to_sentence(tgt_out[0].tolist())}")
        break
