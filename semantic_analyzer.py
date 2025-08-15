#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义相似度分析模块
使用sentence-transformers计算语义相似度
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import faiss
import pickle
import os

class SemanticAnalyzer:
    """语义相似度分析器类"""
    
    def __init__(self, config: Dict):
        """
        初始化语义分析器
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.medical_vectors = None
        self.medical_terms = []
        self.faiss_index = None
        
        # 向量缓存文件路径
        self.vector_cache_file = "medical_vectors.pkl"
        
    def load_model(self):
        """加载sentence-transformers模型"""
        try:
            print("正在加载语义相似度模型...")
            # 使用多语言模型，支持中文
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保已安装sentence-transformers并且网络连接正常")
            raise
    
    def build_medical_corpus(self, drugs: List[str], devices: List[str], 
                           diseases: List[str], hospitals: List[str], samples: List[str] = None):
        """
        构建医疗术语语料库
        Args:
            drugs: 药物术语列表
            devices: 医疗器械术语列表  
            diseases: 疾病术语列表
            hospitals: 医院术语列表
            samples: 样本术语列表（可选，仅用于语义分析器初始化）
        """
        # 合并所有医疗术语
        self.medical_terms = drugs + devices + diseases + hospitals
        
        # 如果提供了样本术语，也添加到语料库中
        if samples:
            self.medical_terms.extend(samples)
        
        # 添加一些医疗相关的语境句子，提高语义理解
        medical_context = [
            "皮肤病治疗", "白癜风症状", "银屑病用药", "激光治疗仪器",
            "皮肤科医院", "药物治疗", "医疗器械", "病症诊断",
            "治疗方法", "医疗咨询", "皮肤护理", "疾病预防"
        ]
        
        self.medical_terms.extend(medical_context)
        
    def encode_medical_terms(self):
        """
        将医疗术语编码为向量:medical_vectors.pkl
        一，如果存在缓存文件则直接加载，然后比对术语是否一致，
        二，否则重新计算并缓存，构建FAISS索引用于快速搜索
        """
        if os.path.exists(self.vector_cache_file):
            print("发现向量缓存文件，正在加载...")
            try:
                # self.vector_cache_file是路径的字符串
                # 'rb' 模式
                # r：表示以只读方式打开文件
                # b：表示以二进制模式读取（因为pickle序列化的数据是二进制格式）
                # 必须用二进制模式，否则会报编码错误
                with open(self.vector_cache_file, 'rb') as f:
                    # Python的pickle模块用于序列化和反序列化对象。这里将二进制文件内容反序列化为原始的Python对象（通常是字典或列表）。
                    cache_data = pickle.load(f)
                    self.medical_vectors = cache_data['vectors']
                    cached_terms = cache_data['terms']
                    
                    # 检查缓存的术语是否与当前术语一致
                    if set(cached_terms) == set(self.medical_terms):
                        print("向量缓存加载成功！")
                        self._build_faiss_index()
                        return
                    else:
                        print("术语已更新，重新计算向量...")
            except Exception as e:
                print(f"加载缓存失败: {e}，重新计算向量...")
        
        print("正在计算医疗术语向量（首次运行较慢，请耐心等待）...")
        
        if self.model is None:
            self.load_model()
            
        # 批量编码，提高效率

        # 参数签名匹配：通过参数名和数量匹配最接近的方法
        self.medical_vectors = self.model.encode(
            self.medical_terms, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # 缓存向量到文件
        try:
            # cache_data 是字典
            cache_data = {
                'vectors': self.medical_vectors,
                'terms': self.medical_terms
            }
            # 'wb'：以二进制写入模式打开文件（w=写，b=二进制）
            with open(self.vector_cache_file, 'wb') as f:

                # 作用：将Python对象序列化为二进制数据并写入文件。
                # 机制：
                # 将 cache_data 字典转换为字节流
                # 写入到打开的文件对象 f 中
                pickle.dump(cache_data, f)
            print("向量已缓存到文件！")
        except Exception as e:
            print(f"缓存向量失败: {e}")
        
        # 构建FAISS索引用于快速搜索
        self._build_faiss_index()
        
    def _build_faiss_index(self):
        """构建FAISS索引用于快速相似度搜索"""
        if self.medical_vectors is None:
            return
            
        # 创建FAISS索引
        dimension = self.medical_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)   #  初始化索引时必须指定维度 # IP意思为 使用内积（余弦相似度）
        
        # 将向量归一化（用于余弦相似度计算）
        # v = np.array([1, 2, 3])
        # norm = np.linalg.norm(v)  # 计算范数：√(1²+2²+3²) ≈ 3.741
        # normalized_v = v / norm   # 得到 [0.267, 0.534, 0.802]
        # 归一化：将向量转换为单位向量，使其长度为1。
        normalized_vectors = self.medical_vectors / np.linalg.norm(
            self.medical_vectors, axis=1, keepdims=True
        )
        
        # 添加向量到索引
        # faiss_index.add不会逐条遍历添加向量，而是以批量矩阵操作的方式一次性高效添加所有向量
        self.faiss_index.add(normalized_vectors.astype('float32'))
        print("FAISS索引构建完成！")
    
    def calculate_similarity(self, sentence: str) -> float:
        """
        计算句子与医疗领域的语义相似度
        Args:
            sentence: 输入句子
        Returns:
            相似度分数（0-1之间）
        """
        if self.model is None:
            self.load_model()
            
        if self.medical_vectors is None:
            print("警告：医疗术语向量未初始化，返回默认相似度0.0")
            return 0.0
        
        # 编码输入句子
        sentence_vector = self.model.encode([sentence])
        
        # 归一化
        sentence_vector = sentence_vector / np.linalg.norm(sentence_vector)
        
        # 使用FAISS搜索最相似的术语
        if self.faiss_index is not None:
            similarities, indices = self.faiss_index.search(
                sentence_vector.astype('float32'), k=5  # 返回前5个最相似的
            )
            
            # 返回最高相似度
            # 例：
            # similarities = [[0.95, 0.82, 0.76, 0.65, 0.58]]  # 相似度从高到低
            # indices      = [[123, 456, 789, 101, 202]]       # 对应的术语索引
            max_similarity = float(similarities[0][0]) if len(similarities[0]) > 0 else 0.0
            return max(0.0, min(1.0, max_similarity))  # 确保在0-1范围内
        else:
            # 如果没有FAISS索引，使用传统方法计算
            # sentence_vector：是你当前要查询的句子向量（shape 一般是 (1, d) 或 (d,)）。
            # self.medical_vectors：是你事先存好的所有医学相关文本的向量（shape 一般是 (n, d)）。
            # self.medical_vectors.T：把 (n, d) 转置成 (d, n)，这样才能和 (1, d) 做矩阵乘法。
            # 数学上就是把当前句子向量和每个医学向量分别做 点积（inner product）。
            # similarities是和每个医学向量做点积的结果，shape是(1, n)
            similarities = np.dot(sentence_vector, self.medical_vectors.T)
            max_similarity = float(np.max(similarities))
            return max(0.0, min(1.0, max_similarity))
    
    def find_most_similar_terms(self, sentence: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        找到与句子最相似的医疗术语
        Args:
            sentence: 输入句子
            top_k: 返回前k个最相似的术语
        Returns:
            相似术语列表，格式为[(术语, 相似度), ...]
        """
        if self.model is None or self.medical_vectors is None:
            return []
            
        # 编码输入句子
        sentence_vector = self.model.encode([sentence])
        sentence_vector = sentence_vector / np.linalg.norm(sentence_vector)
        
        if self.faiss_index is not None:
            # 使用FAISS搜索
            similarities, indices = self.faiss_index.search(
                sentence_vector.astype('float32'), k=top_k
            )
            
            results = []
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.medical_terms):  # 确保索引有效
                    term = self.medical_terms[idx]
                    results.append((term, float(sim)))
            
            return results
        else:
            # 传统方法
            similarities = np.dot(sentence_vector, self.medical_vectors.T)[0]
            
            # 获取前k个最相似的索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                term = self.medical_terms[idx]
                sim = float(similarities[idx])
                results.append((term, sim))
            
            return results