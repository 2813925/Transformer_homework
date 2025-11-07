"""
处理IWSLT2017下载的XML文件
从de-en.zip解压后的文件中提取训练/验证/测试数据
"""
import os
import xml.etree.ElementTree as ET
import re


def clean_text(text):
    """清理文本"""
    if not text:
        return ""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_xml(xml_file):
    """从IWSLT XML文件提取文本"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        sentences = []
        # 尝试不同的XML结构
        for seg in root.iter('seg'):
            if seg.text:
                text = clean_text(seg.text)
                if text:
                    sentences.append(text.lower())
        
        # 如果没找到seg标签，尝试其他标签
        if not sentences:
            for elem in root.iter():
                if elem.text and elem.tag not in ['srcset', 'doc', 'tstset', 'refset']:
                    text = clean_text(elem.text)
                    if text and len(text.split()) > 1:
                        sentences.append(text.lower())
        
        return sentences
    except Exception as e:
        print(f"处理 {xml_file} 时出错: {e}")
        return []


def extract_text_from_tags_file(tags_file):
    """从train.tags文件提取文本（去除XML标签）"""
    sentences = []
    try:
        with open(tags_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 跳过XML标签行
                if line.startswith('<') and line.endswith('>'):
                    continue
                
                # 移除行内的XML标签
                text = re.sub(r'<[^>]+>', '', line)
                text = clean_text(text)
                
                if text and len(text.split()) > 0:
                    sentences.append(text.lower())
        
        return sentences
    except Exception as e:
        print(f"处理 {tags_file} 时出错: {e}")
        return []


def process_iwslt_data(source_dir='de-en', output_dir='data'):
    """
    处理IWSLT数据文件
    
    参数:
        source_dir: 解压后的de-en目录路径
        output_dir: 输出目录
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("IWSLT2017 数据处理")
    print("=" * 80)
    
    # 文件映射
    file_mapping = {
        'train': {
            'en': 'train.tags.de-en.en',
            'de': 'train.tags.de-en.de'
        },
        'valid': {
            'en': 'IWSLT17.TED.dev2010.de-en.en.xml',
            'de': 'IWSLT17.TED.dev2010.de-en.de.xml'
        },
        'test': {
            'en': 'IWSLT17.TED.tst2015.de-en.en.xml',
            'de': 'IWSLT17.TED.tst2015.de-en.de.xml'
        }
    }
    
    results = {}
    
    for split, files in file_mapping.items():
        print(f"\n处理 {split} 数据...")
        
        en_file = os.path.join(source_dir, files['en'])
        de_file = os.path.join(source_dir, files['de'])
        
        # 检查文件是否存在
        if not os.path.exists(en_file):
            print(f"  ⚠️  找不到英语文件: {en_file}")
            continue
        if not os.path.exists(de_file):
            print(f"  ⚠️  找不到德语文件: {de_file}")
            continue
        
        # 提取文本
        if split == 'train':
            # 训练集是tags格式
            en_sentences = extract_text_from_tags_file(en_file)
            de_sentences = extract_text_from_tags_file(de_file)
        else:
            # 验证集和测试集是XML格式
            en_sentences = extract_text_from_xml(en_file)
            de_sentences = extract_text_from_xml(de_file)
        
        print(f"  英语句子: {len(en_sentences)}")
        print(f"  德语句子: {len(de_sentences)}")
        
        # 确保句子数量匹配
        min_len = min(len(en_sentences), len(de_sentences))
        if len(en_sentences) != len(de_sentences):
            print(f"  ⚠️  警告: 英德句子数量不匹配，使用最小长度: {min_len}")
        
        en_sentences = en_sentences[:min_len]
        de_sentences = de_sentences[:min_len]
        
        # 保存到文件
        en_output = os.path.join(output_dir, f'{split}.en')
        de_output = os.path.join(output_dir, f'{split}.de')
        
        with open(en_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(en_sentences) + '\n')
        
        with open(de_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(de_sentences) + '\n')
        
        print(f"  ✅ 已保存: {en_output} ({min_len} 行)")
        print(f"  ✅ 已保存: {de_output} ({min_len} 行)")
        
        results[split] = min_len
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("处理完成！数据统计:")
    print("=" * 80)
    
    total = 0
    for split in ['train', 'valid', 'test']:
        if split in results:
            count = results[split]
            total += count
            
            en_file = os.path.join(output_dir, f'{split}.en')
            de_file = os.path.join(output_dir, f'{split}.de')
            
            en_size = os.path.getsize(en_file) / (1024 * 1024)
            de_size = os.path.getsize(de_file) / (1024 * 1024)
            
            print(f"{split:10s}: {count:8,} 句对")
            print(f"            EN: {en_size:6.2f} MB")
            print(f"            DE: {de_size:6.2f} MB")
    
    print(f"{'总计':10s}: {total:8,} 句对")
    print("=" * 80)
    
    # 显示样例
    print("\n数据样例:")
    print("-" * 80)
    for split in ['train', 'valid', 'test']:
        if split in results:
            en_file = os.path.join(output_dir, f'{split}.en')
            de_file = os.path.join(output_dir, f'{split}.de')
            
            with open(en_file, 'r', encoding='utf-8') as f_en, \
                 open(de_file, 'r', encoding='utf-8') as f_de:
                en_line = f_en.readline().strip()
                de_line = f_de.readline().strip()
                
                print(f"\n{split} 样例:")
                print(f"  EN: {en_line[:80]}...")
                print(f"  DE: {de_line[:80]}...")
    
    print("\n" + "=" * 80)
    print("✅ 数据处理完成！可以开始训练了。")
    print("=" * 80)
    
    return results


def main():
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    else:
        source_dir = 'de-en'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'data'
    
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    if not os.path.exists(source_dir):
        print(f"❌ 错误: 找不到源目录 '{source_dir}'")
        print("\n使用方法:")
        print("  python process_iwslt_data.py [源目录] [输出目录]")
        print("\n示例:")
        print("  python process_iwslt_data.py de-en data")
        print("  python process_iwslt_data.py /path/to/de-en /path/to/output")
        return
    
    try:
        process_iwslt_data(source_dir, output_dir)
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
