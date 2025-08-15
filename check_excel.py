#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查Excel输出结果的脚本
"""
import pandas as pd

def check_excel_output(excel_file: str):
    """检查Excel文件的内容"""
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        
        print(f"=== Excel文件内容检查: {excel_file} ===")
        print(f"总行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head().to_string(index=False))
        
        print("\n数据类型:")
        print(df.dtypes)
        
        # 统计相关度分布
        if '相关度' in df.columns:
            print(f"\n相关度统计:")
            high_relevance = len(df[df['相关度'] >= 0.8])
            medium_relevance = len(df[(df['相关度'] >= 0.5) & (df['相关度'] < 0.8)])
            low_relevance = len(df[df['相关度'] < 0.5])
            
            print(f"高相关度 (≥0.8): {high_relevance} 条")
            print(f"中等相关度 (0.5-0.8): {medium_relevance} 条")  
            print(f"低相关度 (<0.5): {low_relevance} 条")
            print(f"平均相关度: {df['相关度'].mean():.3f}")
        
        # 检查关键术语分布
        if '关键术语' in df.columns:
            print(f"\n关键术语统计:")
            non_empty_terms = len(df[df['关键术语'].str.len() > 0])
            print(f"有关键术语: {non_empty_terms} 条")
            print(f"无关键术语: {len(df) - non_empty_terms} 条")
            
            if non_empty_terms > 0:
                print("\n关键术语频次:")
                term_counts = df[df['关键术语'].str.len() > 0]['关键术语'].value_counts()
                print(term_counts.head(10).to_string())
        
        print("\n✓ Excel文件格式正确！")
        
    except Exception as e:
        print(f"检查Excel文件时出错: {e}")

if __name__ == "__main__":
    check_excel_output("test_sentences_analyzed.xlsx")