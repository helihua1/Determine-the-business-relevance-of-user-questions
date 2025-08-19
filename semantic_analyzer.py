#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义相似度分析模块
使用sentence-transformers计算语义相似度
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample, losses
from typing import List, Dict, Tuple
import faiss
import pickle
import os
from torch.utils.data import DataLoader
import random
import torch  # 添加torch导入


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
        self.device = None  # 添加设备属性
        
        # 向量缓存文件路径
        self.vector_cache_file = "medical_vectors.pkl"
        
        # 模型权重缓存路径
        self.model_weights_dir = "trained_model_weights"
        
        # 初始化时检测设备
        self._detect_device()
        
    def _detect_device(self):
        """检测并设置可用的设备（GPU或CPU）"""
        try:
            if torch.cuda.is_available():
                # 检查GPU内存状态
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                
                print(f"检测到GPU: {torch.cuda.get_device_name()}")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU总内存: {gpu_memory:.2f} GB")
                print(f"GPU可用内存: {free_memory:.2f} GB")
                
                # 如果GPU内存不足，尝试清理
                if free_memory < 1.0:  # 小于1GB可用内存
                    print("GPU内存不足，正在清理...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # 重新检查内存
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                    print(f"清理后GPU可用内存: {free_memory:.2f} GB")
                
                # 测试GPU是否真的可用
                try:
                    test_tensor = torch.randn(100, 100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    print("GPU测试成功，使用GPU")
                    self.device = torch.device('cuda')
                except Exception as e:
                    print(f"GPU测试失败: {e}")
                    print("切换到CPU模式")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
                print("未检测到CUDA，使用CPU")
        except Exception as e:
            print(f"设备检测出错: {e}")
            print("使用CPU作为备选方案")
            self.device = torch.device('cpu')
    
    def _load_trained_weights(self) -> bool:
        """
        尝试加载已训练的模型权重
        Returns:
            bool: 是否成功加载权重
        """
        try:
            # 检查权重目录是否存在
            if not os.path.exists(self.model_weights_dir):
                print(f"权重目录不存在: {self.model_weights_dir}")
                return False
            
            # 检查权重文件是否存在
            weights_path = os.path.join(self.model_weights_dir, "model_weights.pth")
            if not os.path.exists(weights_path):
                print(f"权重文件不存在: {weights_path}")
                return False
            
            # 检查权重文件是否完整（至少1MB）
            if os.path.getsize(weights_path) < 1024 * 1024:
                print("权重文件过小，可能损坏")
                return False
            
            print(f"发现权重文件: {weights_path}")
            
            # 加载预训练模型
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # 将模型移动到检测到的设备
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
            
            # 加载训练后的权重
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            print("模型权重加载成功！")
            return True
            
        except Exception as e:
            print(f"加载权重失败: {e}")
            return False
    
    def _save_trained_weights(self):
        """保存训练后的模型权重"""
        try:
            # 创建权重目录
            if not os.path.exists(self.model_weights_dir):
                os.makedirs(self.model_weights_dir)
            
            # 保存权重
            weights_path = os.path.join(self.model_weights_dir, "model_weights.pth")
            state_dict = self.model.state_dict()
            torch.save(state_dict, weights_path)
            
            print(f"模型权重已保存到: {weights_path}")
            
        except Exception as e:
            print(f"保存权重失败: {e}")
    
    def _should_retrain(self) -> bool:
        """
        检查是否需要重新训练模型
        Returns:
            bool: 是否需要重新训练
        """
        try:
            # 检查权重文件是否存在
            weights_path = os.path.join(self.model_weights_dir, "model_weights.pth")
            if not os.path.exists(weights_path):
                print("权重文件不存在，需要训练")
                return True
            
            # 检查权重文件是否完整
            if os.path.getsize(weights_path) < 1024 * 1024:
                print("权重文件损坏，需要重新训练")
                return True
            
            # 检查向量缓存是否存在且完整
            if not os.path.exists(self.vector_cache_file):
                print("向量缓存不存在，需要训练")
                return True
            
            # 检查向量缓存文件大小
            if os.path.getsize(self.vector_cache_file) < 1024 * 100:  # 至少100KB
                print("向量缓存文件过小，需要重新训练")
                return True
            
            print("所有缓存文件完整，无需重新训练")
            return False
            
        except Exception as e:
            print(f"检查训练状态时出错: {e}")
            return True
    
    def force_retrain(self):
        """
        强制重新训练模型
        删除所有缓存文件并重新训练
        """
        try:
            print("强制重新训练模型...")
            
            # 删除权重文件
            weights_path = os.path.join(self.model_weights_dir, "model_weights.pth")
            if os.path.exists(weights_path):
                os.remove(weights_path)
                print(f"已删除权重文件: {weights_path}")
            
            # 删除向量缓存文件
            if os.path.exists(self.vector_cache_file):
                os.remove(self.vector_cache_file)
                print(f"已删除向量缓存文件: {self.vector_cache_file}")
            
            # 删除权重目录（如果为空）
            if os.path.exists(self.model_weights_dir) and not os.listdir(self.model_weights_dir):
                os.rmdir(self.model_weights_dir)
                print(f"已删除空权重目录: {self.model_weights_dir}")
            
            print("缓存清理完成，下次启动时将重新训练")
            
        except Exception as e:
            print(f"清理缓存失败: {e}")
        
    def load_model(self):
        """加载sentence-transformers模型"""
        try:
            print("正在加载语义相似度模型...")
            
            # 检查是否需要重新训练
            if not self._should_retrain():
                # 如果不需要重新训练，尝试加载已训练的模型权重
                if self._load_trained_weights():
                    print("成功加载已训练的模型权重！")
                    return
                else:
                    print("权重加载失败，将重新训练...")
            
            # 需要训练或权重加载失败，加载预训练模型
            print("加载预训练模型...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # 将模型移动到检测到的设备
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
                print(f"模型已移动到GPU: {torch.cuda.get_device_name()}")
            else:
                print("模型使用CPU运行")
                
            print("模型加载完成！")

            # 加载完成后训练模型
            self.train_model()
            
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
        # 根据设备类型设置batch_size
        batch_size = 64 if self.device.type == 'cuda' else 32
        
        # 参数签名匹配：通过参数名和数量匹配最接近的方法
        self.medical_vectors = self.model.encode(
            self.medical_terms, 
            show_progress_bar=True,
            batch_size=batch_size,
            device=self.device  # 指定设备
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
        sentence_vector = self.model.encode([sentence], device=self.device)
        
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

            # max_similarity = float(similarities[0][0]) if len(similarities[0]) > 0 else 0.0
            # return max(0.0, min(1.0, max_similarity))  # 确保在0-1范围内
            max_similarity = float(similarities[0][0])
            return max_similarity
        else:
            # 如果没有FAISS索引，使用传统方法计算
            # sentence_vector：是你当前要查询的句子向量（shape 一般是 (1, d) 或 (d,)）。
            # self.medical_vectors：是你事先存好的所有医学相关文本的向量（shape 一般是 (n, d)）。
            # self.medical_vectors.T：把 (n, d) 转置成 (d, n)，这样才能和 (1, d) 做矩阵乘法。
            # 数学上就是把当前句子向量和每个医学向量分别做 点积（inner product）。
            # similarities是和每个医学向量做点积的结果，shape是(1, n)
            print("如果没有FAISS索引，使用传统方法计算")
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
        sentence_vector = self.model.encode([sentence], device=self.device)
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
    
    def train_model(self):
        """
        训练语义相似度模型
        Args:
            sentences: 输入句子列表
            labels: 标签列表（0或1）
        """
        baidianfeng = ["白癜风","白碘风","白殿风","白斑风","白颠风","白癫风","白点癫风","白屑风","白瘕风","白颠疯","白电风","白痶风","白疒癜风","白癞癜风","白电疯","百点风","百癜疯","白巅风","白垫风","白瘨风","白店风","白巅峰","白殿凤","白典风","白点风","白广癜风","白殿疯","白殿","自殿风","白癜凤","白癫","白癫疯","白疯癫","白癲风","白驳风","白癞风","白点癫疯","白点癫","白瘢风","白癜","白颠","白班癫疯","白癣风","白壂风","百瘕风","白淀风","白癜疯","白班风","白斑疯","白癬风","白班疯","白巅疯","白蚀病","白斑病","白风病","白麻风","白皮风","白斑症","白风癣","白块风","白蚀","白驳","皮肤白斑","白癞","白秃风","白蚀风","色素脱失性白斑病","白班病","色素脱失症","色素脱落症","色素脱失","白头风","白蚀症","白驳症","白电凤病","白佃风","白滇风","白掂风","白癫病","白颠病","白点疯","白癫症","白斑性癫风","白点颠风","白点癜风","白点颠疯","白点癜疯","色素减退斑","晕痣","白点1癫风","白瘾风","白疯颠","白癖风","白点一癫风","白癍风","白风点癫","白点癞风","白点癜","白点颠","白点巅峰","白淀疯","白壂疯","白爹风","白跌风","白瘹风","白跌疯","白叠风","白丁风","白蝶疯","白蝶风","白丁疯","白鼎疯","白顶风","白定疯","白腚风","白风癫","自癜风","白臀风","白痴风","白疒癫疯","白病癫风","白病癫","白颤风","白厂颠风","白颤疯","白瘨疯","白巅病","白巅凤","白典疯","白定风","白鼎风","白风癜","白片风","白天风","白腆风","白广癫疯","白瘕疯","白疯疯病","白痹风","白色癫疯","白厩风","白脸风","晕痔","白颠风","白癫疯","白廒风","白瘢疯","白瘢凤","白瘢冈","白癍病","白癍疯","白壁风","白臂风","白边风","白编风","白鞭风","白鞭疯","白扁风","白变风","白变疯","白变凤","白便风","白遍风","白飙风","白瘭风","白鬓风","炎症色素减退白斑","炎症性白斑","炎症后色素减退白斑","炎症后白斑","炎症色素","色素减退","白㿄风","白搬风","白陛风","白陛疯","白廦风","白避风","白璧风","白边疯","白边峰","白边锋","白边凤","白便疯","白遍疯","白瘪风","白疒殿风","白疒疯","白病风","白病疯","白波风","白玻疯","白博疯","白簸风","白廍风","白藏风","白痴癜风","白痴疯疯","白痴疯","白疵风","白瘯风","白戴风","白戴疯","白瘅风","白得风","白倒风","白癖疯","白登风","白登疯","白滴风","白底疯","白弟风","白帝癫风","白帝癜风","白缔疯","白嗲风","白掂疯","白滇疯","白滇峰","白瘨","白癲病","白碘疯","白典凤","白电癫风","白电癫","白电凤","白甸疯","白店枫","白店疯","白店凤","白垫疯","白叠疯","白碟风","白顶疯","白嵿疯","白订疯","白段风","白额风","白风颠","白风殿","白疯病","白疯巅","白疯淀","白疯殿","白疯癜","白疯风","白峰巅","白广癫风","白广癫冈","白广殿风","白广癜","白户癫疯","白痪风","白癀风","白毁风","白屐风","白箕风","白见风","白见疯","白建风","白贱风","白剑风","白健风","白厩疯","白就风","白廄风","白厥风","白连风","白利风","白连疯","白连凤","白瘤风","白癃风","白瘰风","白屡疯","白履风","白瘼风","白瘼疯","白内风","白劈风","白譬风","白偏风","白偏疯","白偏锋","白篇风","白翩风","白谝风","白片疯","白颇风","白颇疯","白瘦风","白厮风","白瘶风","白瘫风","白天癜风","白填风","白填疯","白阗疯","白痶疯","白鲜风","白鲜疯","白廯风","白痫风","白痫疯","白显风","白线风","白消风","白疫癜风","白殷风","白瘀风","白展风","白瘵风","白展殿风","白珍风","白真殿风","百瘢风","百边风","百扁风","百痴风","百颠风","百颠疯","百巅风","百巅峰","百癫风","百癫疯","百癫凤","百典风","百典疯","百点疯","百电风","百店风","百垫风","百淀风","百淀疯","百殿风","百殿疯","百殿凤","百癜风","百丁风","百丁疯","百顶风","百顶疯","百定风","百定疯","自颠风","自瘢风","自变风","自颠疯","自巅风","自巅疯","自巅峰","自癫风","自癫疯","自点风","自殿疯","自电风","自店风","自点疯","自淀风","自癜疯","自丁风","自瘕风","癜风","百巅疯","白带白癜风","白带癜风","白带疯","痞白","痞白症","龙舐","白顶峰","白瘕病","白捡风","白尖锋","白经风","白连峰","白莲风","白莲疯","白散风","白天癫风","白尉风","百变风","白厢风","白旋风","白丹病","白帝风","白底风","白堤风","白的风","白地风","白奌风","白电峰","白淀病","白皮病","斑驳病","白厨风","白癣症","白虎风","白痁疯","白巅","白颊疯","白驳病","白廯","白面疯"]
        yinxiebing = ["银屑病","牛皮皮癣","牛皮肤癣","牛皮癣","牛批癣","牛脾癣","牛疲癣","牛匹癣","牛辟癣","牛皮鲜","牛皮显","牛皮线","牛皮仙","牛皮轩","牛批藓","牛皮癖","牛皮藓","牛皮痟","牛皮廨","白疕","牛疲藓","牛痞藓","牛癖鲜","牛屁藓","牛藓病","牛屑癣","牛癣图","牛皮广癣","牛皮廯","银屑癣","银消病","银痟病","银削病","银宵病","银血病","牛藓","牛癣","牛银","牛皮病癣","副银病","白银屑","负银屑","副银屑","牛鼻癣","牛痹癣","牛血癣","银光屑","银鳞病","银皮病","银皮屑","银皮癣","银皮症","银屏癣","银钱癣","银悄病","银翘病","银翘藓","银鞘病","银翘癣","银锡病","银消癣","银消症","银绡癣","银硝藓","银销癣","银霄病","银肖病","银屑斑","银屑点","银屑风","银屑疾","银屑甲","银屑鲜","银屑廯","银屑藓","银屑屑","银屑血","银屑炎","银屑症","银癣病","银雪癣","银血癣","银血症","长银屑","滴状银屑","点滴银病","点滴银屑","点状银屑","副银屑症","牛皮斑癣","牛皮恶癣","牛皮康癣","牛皮皮藓","牛皮头癣","牛皮顽癣","牛皮小癣","牛皮屑癣","牛皮选癣","牛皮血癣","牛皮一癣","牛皮银屑","牛皮之癣","牛钱癣病","银病","银肤癣","银屑牛癣","银屑皮癣","牛皮有癣","牛皮初期癣","银皮肤癣","银皮肤病","银宵湿","银币癣","银销病","银肖疾","银血屑","银癣","牛皮癬","牛皮屑","牛皮病","牛皮消","牛斑癣","牛股藓","牛肉癣","银宵癣","银皮藓","银俏病","银头皮屑","银藓","银硝病","银线病","银癣苪","银削","银雪病","银元癣","白银癣","牛反癣","牛皮选","牛皮先","牛皮血","牛钱藓","牛屁癣","俞银皮","银鳞屑","银泻病","牛皮好癣","牛皮荃","牛皮想","牛皮炎","牛气癣","牛有癣","银前癣","牛皮蘚","午皮癣","牛皮瘢","牛皮m癣","牛皮痒","牜皮癣","银宵疯","牛皮顽屑","牛皮皮肤癣","银消痛","银嘱病","牛皮虫","牛皮的癣","牛皮个癣","银綃病","银肖","银消","皮廯","皮藓","皮屑","皮癣","屁藓","屁癣","生藓","生癣","湿藓","湿疹","手廯","手藓","手癣","体藓","体屑","体癣","头鲜","头廯","头癣","腿藓","腿癣","癣疹","长廯","长癣","掌脓包","掌脓疱","掌拓病","掌跖病","掌跖脓","掌跖症","扁平苔藓","扁平苔癣","红藓","红癣","红皮病","红皮型","副银屑病","银屑病甲"]
        wuguanbing = ["皮炎","头皮真菌感染","荨麻疹","肝","降糖","高血压","咬","高血脂","糖尿病","湿疹","肛瘘","肾","祛斑","高血压","脑病","颈椎","癫痫","鼻炎","抑郁症","中风","心脏病","腰突","腰椎","肠胃炎","便秘","淋巴结核","癌","胃病","肿瘤","皮炎","黄褐斑","去斑","美容","湿疹","疱疹","斑秃","玫瑰痤疮","白内障","荨麻疹","脱发","痘痘","脚气","红斑狼疮","白色糠疹","致畸","双休","唇炎","脚气","紫癜","鱼鳞病","鼻喷","尿布疹","痤疮","唇炎","痔疮","扁平疣","粉刺","泡疹","扁平苔藓","口疮","玫瑰糠疹","白色糠疹","囊肿","突眼","眼凸","普秃","白塞病","强直性","灰指甲","火疖","祛痘","疤痕","猫癣","头癣","祛疤","毛囊炎","狐臭","痱子","糠疹","虫咬","脂溢性","疣","花斑癣","脚臭","肾病","疮","痘","口周炎","脚癣","狐臭","酒渣鼻","口角炎","硬皮病","肿瘤","泌尿","白内障","白血病","白念珠菌","白塞病","眉毛","头发","白发"] 
        combined = baidianfeng + yinxiebing

        # #因为loss过高，8左右，所以放弃
        # # 会将相同texts中识别为相似，其他texts中识别为不相似
        # train_examples = [
        #     InputExample(texts= combined),   # 正样本
        #     InputExample(texts= wuguanbing)   # 正样本(自动被combined互相识别为无关)
        # ]
        #
        # # 创建训练数据加载器，用于批量处理训练样本
        # # shuffle=True: 随机打乱训练数据顺序，提高训练效果
        # # batch_size=16: 每批处理16个样本，平衡内存使用和训练效率
        # # pin_memory: 根据设备类型动态设置，GPU时启用以加速数据传输
        # pin_memory = self.device.type == 'cuda'
        # train_dataloader = DataLoader(
        #     train_examples,
        #     shuffle=True,
        #     batch_size=16,
        #     pin_memory=pin_memory
        # )
        #
        # # 定义训练损失函数：多负样本排序损失
        # train_loss = losses.MultipleNegativesRankingLoss(self.model)
        #
        # # 开始训练模型
        # print("开始MultipleNegativesRankingLoss训练！")
        # self.model.fit(
        #     # train_objectives: 训练目标列表，包含(数据加载器, 损失函数)的元组
        #     # 这里只有一个训练目标，也可以设置多个不同的训练任务
        #     train_objectives=[(train_dataloader, train_loss)],
        #     # epochs=2: 训练轮数，模型将整个训练数据集学习2遍
        #     epochs=1,
        #     show_progress_bar=True,
        #     # GPU优化参数
        #     optimizer_params={'lr': 2e-5} if self.device.type == 'cuda' else {}
        # )
        
        # 1. 创建正样本对：相同疾病类别内的别名配对
        positive_pairs = []
        for disease_list in [baidianfeng, yinxiebing]:
            for i in range(len(disease_list)):
                for j in range(i + 1, min(i + 5, len(disease_list))):  # 每个词与后续4个词配对
                    positive_pairs.append((disease_list[i], disease_list[j]))

        # 2. 创建负样本对：不同疾病类别间的配对
        negative_pairs = []
        for i in range(2000):  # 创建2000个负样本对
            # 白癜风和银屑病 vs 无关疾病
            negative_pairs.append((random.choice(combined), random.choice(wuguanbing)))


        # 3. 创建训练样本
        train_examples = []
        # 添加正样本
        for pair in positive_pairs:
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=1.0))
        # 添加负样本
        for pair in negative_pairs:
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=0.0))

        # 4. 创建训练数据加载器
        # 根据设备类型设置pin_memory参数
        pin_memory = self.device.type == 'cuda'  # 只有GPU时才启用pin_memory
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=16,
            pin_memory=pin_memory  # 动态设置pin_memory
        )

        # 5. 使用更适合的损失函数 - 对比损失更适合这种二元分类
        from sentence_transformers.losses import ContrastiveLoss
        train_loss = ContrastiveLoss(model=self.model)

        # 6. 开始训练
        print("开始训练！")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=5,  # 增加训练轮数
            show_progress_bar=True,
            # evaluator=None,  # 可以添加评估器
            # evaluation_steps=100,  # 每100步评估一次
            optimizer_params={'lr': 2e-5}  # 设置学习率
        )

        # 训练完成后保存模型权重
        print("训练完成，正在保存模型权重...")
        self._save_trained_weights()