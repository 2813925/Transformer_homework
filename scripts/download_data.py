"""
IWSLT2017 (EN-DE) 数据集下载脚本
Machine Translation: English to German
"""
import os
from datasets import load_dataset


def download_iwslt2017(save_dir='data'):
    """下载IWSLT2017 EN-DE数据集"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("正在下载IWSLT2017 (EN->DE) 数据集...")
    print("数据集: Machine Translation (English to German)")
    print("约200K句对")
    
    try:
        # 从Hugging Face下载数据集
        dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')
        
        print("下载完成，正在保存...")
        
        # 保存训练集
        train_data = dataset['train']
        with open(os.path.join(save_dir, 'train.en'), 'w', encoding='utf-8') as f_en:
            with open(os.path.join(save_dir, 'train.de'), 'w', encoding='utf-8') as f_de:
                for example in train_data:
                    en_text = example['translation']['en'].strip()
                    de_text = example['translation']['de'].strip()
                    if en_text and de_text:  # 过滤空行
                        f_en.write(en_text + '\n')
                        f_de.write(de_text + '\n')
        
        # 保存验证集
        valid_data = dataset['validation']
        with open(os.path.join(save_dir, 'valid.en'), 'w', encoding='utf-8') as f_en:
            with open(os.path.join(save_dir, 'valid.de'), 'w', encoding='utf-8') as f_de:
                for example in valid_data:
                    en_text = example['translation']['en'].strip()
                    de_text = example['translation']['de'].strip()
                    if en_text and de_text:
                        f_en.write(en_text + '\n')
                        f_de.write(de_text + '\n')
        
        # 保存测试集
        test_data = dataset['test']
        with open(os.path.join(save_dir, 'test.en'), 'w', encoding='utf-8') as f_en:
            with open(os.path.join(save_dir, 'test.de'), 'w', encoding='utf-8') as f_de:
                for example in test_data:
                    en_text = example['translation']['en'].strip()
                    de_text = example['translation']['de'].strip()
                    if en_text and de_text:
                        f_en.write(en_text + '\n')
                        f_de.write(de_text + '\n')
        
        print(f"\nIWSLT2017数据集已保存到 {save_dir}")
        print("\n数据集文件:")
        
        files = ['train.en', 'train.de', 'valid.en', 'valid.de', 'test.en', 'test.de']
        for filename in files:
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filename}: {lines} 行 ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("\n请确保已安装datasets库:")
        print("pip install datasets")
        print("\n或手动下载数据集:")
        print("访问: https://huggingface.co/datasets/iwslt2017")
        return False


if __name__ == '__main__':
    download_iwslt2017()
