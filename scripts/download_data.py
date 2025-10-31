"""
WikiText-2 数据集下载脚本
"""
import os
import urllib.request
import zipfile


def download_wikitext2(save_dir='data'):
    """下载WikiText-2数据集"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
    zip_path = os.path.join(save_dir, 'wikitext-2.zip')
    
    print("正在下载WikiText-2数据集...")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"下载完成: {zip_path}")
        
        # 解压
        print("正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        
        # 移动文件到正确位置
        extracted_dir = os.path.join(save_dir, 'wikitext-2-raw')
        if os.path.exists(extracted_dir):
            for filename in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, filename)
                dst = os.path.join(save_dir, filename)
                if os.path.isfile(src):
                    os.rename(src, dst)
            os.rmdir(extracted_dir)
        
        # 删除zip文件
        os.remove(zip_path)
        
        print(f"WikiText-2数据集已保存到 {save_dir}")
        print("\n数据集文件:")
        for filename in os.listdir(save_dir):
            if filename.startswith('wiki.'):
                filepath = os.path.join(save_dir, filename)
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filename} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("\n请手动下载数据集:")
        print("1. 访问: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/")
        print("2. 下载 wikitext-2-raw-v1.zip")
        print(f"3. 解压到 {save_dir} 目录")
        return False


if __name__ == '__main__':
    download_wikitext2()
