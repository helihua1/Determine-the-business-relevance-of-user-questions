#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖包安装脚本
运行这个脚本来安装所有必要的依赖包
"""
import subprocess
import sys
import os

def install_requirements():
    """安装requirements.txt中的所有依赖包"""
    try:
        print("正在安装依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
        
        # 下载jieba词典
        print("正在初始化jieba分词...")
        import jieba
        jieba.initialize()
        print("jieba初始化完成！")
        
        # 下载sentence-transformers模型
        print("正在下载语义相似度模型（首次运行需要下载，请耐心等待）...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("模型下载完成！")
        
        print("\n所有依赖安装完成，可以运行主程序了！")
        
    except Exception as e:
        print(f"安装过程中出现错误: {e}")
        print("请检查网络连接或手动安装依赖包")

if __name__ == "__main__":
    install_requirements()