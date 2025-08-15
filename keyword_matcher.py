#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键词匹配模块
包含精确匹配和模糊匹配功能
"""
import jieba
from fuzzywuzzy import fuzz, process
from typing import List, Dict, Tuple, Set
import re

class KeywordMatcher:
    """关键词匹配器类"""
    
    def __init__(self, config: Dict):
        """
        初始化关键词匹配器 - 创建匹配引擎实例
        
        目的: 初始化关键词匹配系统的核心组件和数据结构
        处理逻辑:
        1. 保存系统配置参数
        2. 初始化jieba分词组件
        3. 创建四类术语存储集合（药物、器械、疾病、医院）
        4. 准备统一的术语查找集合
        
        使用技术:
        - jieba.initialize(): 预加载中文分词模型
        - set数据结构: 快速O(1)查找性能
        - 分类存储: 便于类型识别和优先级管理
        
        数据结构:
        - self.drugs: 药物术语集合
        - self.devices: 医疗器械术语集合
        - self.diseases: 疾病术语集合
        - self.hospitals: 医院术语集合
        - self.all_terms: 所有术语的统一集合
        
        Args:
            config: 系统配置字典，包含模糊匹配参数等
        """
        self.config = config
        # 初始化jieba分词
        jieba.initialize()
        
        # 存储不同类型的术语
        self.drugs = set()          # 药物术语
        self.devices = set()        # 医疗器械术语  
        self.diseases = set()       # 疾病术语
        self.hospitals = set()      # 医院术语
        
        # 所有术语的集合（用于快速查找）
        self.all_terms = set()
        
    def load_terms(self, drugs: List[str], devices: List[str], 
                   diseases: List[str], hospitals: List[str]):
        """
        加载各类术语到匹配器 - 将术语库数据加载到匹配引擎
        
        目的: 将从文件加载的术语数据导入匹配系统，优化分词效果
        处理逻辑:
        1. 将列表转换为set集合，提高查找效率
        2. 合并所有术语到统一查找集合
        3. 将术语添加到jieba自定义词典，提高分词准确性
        4. 设置医疗标签和高频权重
        
        使用技术:
        - set集合: O(1)时间复杂度的查找操作
        - jieba.add_word(): 自定义词典扩展
        - 词频权重: freq=1000 提高医疗术语切分优先级
        - 词性标注: tag='medical' 标识医疗术语
        
        性能优化:
        - 使用集合运算(|)快速合并
        - 预添加词典避免运行时重复计算
        
        Args:
            drugs: 药物术语列表，如["雷公藤", "他克莫司"]
            devices: 医疗器械术语列表，如["308激光治疗仪", "光疗仪"]  
            diseases: 疾病术语列表，如["白癜风", "银屑病"]
            hospitals: 医院术语列表，如["宁波鄞州博润皮肤病医院"]
        """
        self.drugs = set(drugs)
        self.devices = set(devices)
        self.diseases = set(diseases)
        self.hospitals = set(hospitals)
        
        # 合并所有术语
        self.all_terms = self.drugs | self.devices | self.diseases | self.hospitals
        
        # 将术语添加到jieba词典中，提高分词准确性
        for term in self.all_terms:
            jieba.add_word(term, freq=1000, tag='medical')
    
    def exact_match(self, sentence: str) -> Set[str]:
        """
        精确匹配 - 在句子中查找完全匹配的医疗术语
        
        目的: 查找句子中完全匹配的医疗术语，确保高准确率
        处理逻辑:
        1. 遍历所有已加载的医疗术语
        2. 使用字符串包含检查(in操作符)进行精确匹配
        3. 将匹配到的术语添加到结果集合
        4. 返回所有精确匹配的术语
        
        使用技术:
        - 字符串包含操作: 'term in sentence' 
        - set集合: 自动去重，快速操作
        - 遍历查找: 简单可靠的匹配策略
        
        匹配特点:
        - 高精度: 完全匹配，无误判
        - 快速: 字符串包含操作效率高
        - 可靠: 不涉及复杂算法，稳定性好
        
        应用场景:
        - "白癜风治疗" → 匹配到 "白癜风"
        - "308激光治疗仪价格" → 匹配到 "308激光治疗仪"
        - "雷公藤效果" → 匹配到 "雷公藤"
        
        Args:
            sentence: 待匹配的输入句子
        Returns:
            Set[str]: 精确匹配到的医疗术语集合
        """
        matched_terms = set()
        
        # 直接字符串匹配
        for term in self.all_terms:
            if term in sentence:
                matched_terms.add(term)
        
        return matched_terms
    
    def fuzzy_match(self, sentence: str) -> List[Tuple[str, int]]:
        """
        模糊匹配 - 使用FuzzyWuzzy算法匹配相似医疗术语
        
        目的: 处理用户输入错误、简写、口语化表达，提高匹配召回率
        处理逻辑:
        1. 使用jieba分词将句子切分为词汇
        2. 过滤掉过短的词汇（长度小于配置的最小长度）
        3. 对每个词汇使用FuzzyWuzzy进行相似度计算
        4. 筛选高于阈值的匹配结果
        5. 去重并按相似度排序返回
        
        使用技术:
        - jieba分词: 中文自然语言处理，提高匹配粒度
        - FuzzyWuzzy.process.extractOne: 模糊字符串匹配
        - fuzz.partial_ratio: 部分匹配算法，适合处理包含关系
        - 编辑距离算法: 计算字符串相似度
        - 阈值过滤: 控制匹配质量
        
        匹配能力:
        - 拼写错误: "雷公藤" ↔ "雷公雷藤"
        - 部分匹配: "宁波博润" ↔ "宁波鄞州博润皮肤病医院" 
        - 简写扩展: "308" ↔ "308激光治疗仪"
        - 同音字错误: 通过编辑距离识别
        
        性能优化:
        - 词长过滤减少无效计算
        - 阈值预筛选避免低质量匹配
        - 去重处理避免重复结果
        
        Args:
            sentence: 待匹配的输入句子
        Returns:
            List[Tuple[str, int]]: 匹配结果列表 [(术语, 相似度分数0-100), ...]
        """
        fuzzy_matches = []
        fuzzy_threshold = self.config['similarity_thresholds']['fuzzy_match']
        min_length = self.config['fuzzy_match']['min_length']
        
        # 对句子进行分词
        words = list(jieba.cut(sentence))
        
        # 对每个分词结果，每个词和all_terms进行模糊匹配，取大于设定分数的那些
        for word in words:
            if len(word) < min_length:#词长阈值过滤
                continue
                
            # 使用process.extractOne找到最相似的术语
            # 例：
            # query = "apple"
            # choices = ["app", "banana", "apples", "pineapple"]
            # best_match = process.extractOne(query, choices)

            # print(best_match)  # 输出: ('apples', 90)

            # 指定匹配算法（scorer=fuzz.partial_ratio），fuzz.partial_ratio 使用编辑距离算法计算字符串相似度

            # 数组l = [1, 2, 3] （array）通常要求 所有元素类型相同（例如 int 数组只能放整数）。
            # 元组t = (1, 2, 3) （tuple）可以放不同类型的数据，元组是 不可变的（immutable），t[0] = 99  # ❌

            # >>> choices = ["苹果", "香蕉", "橘子"]
            # >>> process.extractOne("苹", choices, scorer=fuzz.partial_ratio)
            # 返回 元组：('苹果', 100)
            best_match = process.extractOne(word, self.all_terms, 
                                          scorer=fuzz.partial_ratio)
            # 相似度阈值过滤，（大于设定的阈值）
            if best_match and best_match[1] >= fuzzy_threshold * 100:
                fuzzy_matches.append((best_match[0], best_match[1]))
        
        # 去重并按相似度排序
        unique_matches = {}
        for term, score in fuzzy_matches:
            # term在unique_matches中不存在，如果重复取最高分
            if term not in unique_matches or score > unique_matches[term]:
                unique_matches[term] = score
                
        return [(term, score) for term, score in 
                # 按分数从高到低排序。key=lambda x: x[1]：按元组的第二个元素（score）排序。
                # 通过 items() 将字典转为 (term, score) 元组列表，再通过 key=lambda x: x[1] 指定按分数排序。
                # 等于 sorted([(term, score) for term, score in unique_matches.items()], key=lambda x: x[1], reverse=True)
                sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)]
    
    def get_term_type(self, term: str) -> str:
        """
        获取术语类型
        Args:
            term: 术语
        Returns:
            术语类型：'drug', 'device', 'disease', 'hospital'
        """
        if term in self.drugs:
            return 'drug'
        elif term in self.devices:
            return 'device'
        elif term in self.diseases:
            return 'disease'
        elif term in self.hospitals:
            return 'hospital'
        return 'unknown'
    
    def get_priority_score(self, term: str) -> int:
        """
        获取术语优先级分数（分数越高优先级越高）
        优先级：药物和医疗器械 > 疾病 > 医院
        Args:
            term: 术语
        Returns:
            优先级分数
        """
        term_type = self.get_term_type(term)
        priority_scores = {
            'drug': 100,        # 药物最高优先级
            'device': 100,      # 医疗器械与药物同等优先级  
            'disease': 50,      # 疾病中等优先级
            'hospital': 25      # 医院较低优先级
        }
        return priority_scores.get(term_type, 0)
    
    def find_all_matches(self, sentence: str) -> Dict:
        """
        流程一：关键词匹配
        综合匹配 - 整合精确匹配和模糊匹配的结果，生成统一的匹配报告
        
        目的: 提供完整的术语匹配服务，结合精确和模糊匹配的优势
        处理逻辑:
        1. 执行精确匹配，获得高可信度结果
        2. 执行模糊匹配，获得容错性结果  
        3. 合并两类匹配结果到统一字典
        4. 为每个匹配术语添加元数据（置信度、匹配类型、优先级、术语类型）
        5. 精确匹配结果优先级更高（置信度100）
        
        使用技术:
        - 算法组合: 结合精确和模糊两种匹配策略
        - 元数据管理: 为每个匹配添加丰富的上下文信息
        - 优先级处理: 精确匹配覆盖模糊匹配结果
        - 术语分类: 自动识别药物/器械/疾病/医院类型
        
        返回数据结构:
        {
            '术语名称': {
                'confidence': int,      # 置信度 0-100
                'match_type': str,      # 'exact' 或 'fuzzy'
                'priority': int,        # 业务优先级分数  
                'type': str            # 'drug'/'device'/'disease'/'hospital'
            }
        }
        
        优先级策略:
        - 精确匹配 > 模糊匹配
        - 同术语多次匹配时保留最高质量结果
        - 支持后续的术语提取和相关度计算
        
        Args:
            sentence: 待分析的输入句子
        Returns:
            Dict: 综合匹配结果字典，包含所有匹配术语的详细信息
        """
        # 精确匹配 
        # 看关键词(term)是否在sentence中
        exact_matches = self.exact_match(sentence)
        
        # 模糊匹配 
        # 看sentence中的每个分词后的word和all_terms进行模糊匹配，取大于设定分数的那些
        fuzzy_matches = self.fuzzy_match(sentence)
        
        # 合并结果，精确匹配的优先级更高
        all_matches = {}
        
        # 先添加精确匹配的结果（置信度设为100）
        for term in exact_matches:
            all_matches[term] = {
                'confidence': 100,
                'match_type': 'exact',
                'priority': self.get_priority_score(term),
                'type': self.get_term_type(term)
            }
        
        # 再添加模糊匹配的结果（如果不在精确匹配中）
        for term, score in fuzzy_matches:
            if term not in all_matches:
                all_matches[term] = {
                    'confidence': score,
                    'match_type': 'fuzzy', 
                    'priority': self.get_priority_score(term),
                    'type': self.get_term_type(term)
                }
        
        return all_matches