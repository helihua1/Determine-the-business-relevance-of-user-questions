#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键术语提取器
根据优先级规则提取句子中最关键的术语
优先级：药物和医疗器械 > 疾病 > 医院
"""
from typing import Dict, List, Optional, Tuple

class TermExtractor:
    """术语提取器类"""
    
    def __init__(self):
        """
        初始化术语提取器 - 创建基于业务优先级的术语提取器
        
        目的: 根据医疗业务需求设定术语优先级规则
        处理逻辑:
        1. 定义术语类型的业务优先级映射
        2. 确保药物和医疗器械获得最高优先级
        3. 疾病次之，医院优先级相对较低
        
        使用技术:
        - 优先级映射字典: 数值越大优先级越高
        - 业务规则硬编码: 确保优先级策略的稳定性
        
        业务优先级规则:
        - 药物(drug): 100分 - 核心产品咨询
        - 医疗器械(device): 100分 - 核心产品咨询  
        - 疾病(disease): 50分 - 相关医疗咨询
        - 医院(hospital): 25分 - 地理位置查询
        
        设计思路: 
        优先级反映业务价值，药物和器械是客户核心关注点
        """
        # 术语类型优先级映射（数值越大优先级越高）
        # 未使用
        self.priority_map = {
            'drug': 100,        # 药物最高优先级
            'device': 100,      # 医疗器械与药物同等优先级
            'disease': 50,      # 疾病中等优先级  
            'hospital': 25      # 医院较低优先级
        }
        
    def extract_key_term(self, matched_terms: Dict) -> Optional[str]:
        """
        从匹配的术语中提取最关键的术语 - 核心术语选择算法
        
        目的: 从多个匹配术语中选择最重要、最相关的术语作为输出
        处理逻辑:
        1. 检查输入有效性，空匹配返回None
        2. 调用排序算法对所有术语按重要性排序
        3. 选择排序结果中的第一个术语（最高优先级）
        4. 返回选中的关键术语名称
        
        使用技术:
        - 委托模式: 将排序逻辑委托给专门的排序方法
        - 空值处理: 优雅处理边界情况
        - 单一职责: 只负责选择逻辑，不负责排序逻辑
        
        选择标准:
        - 业务优先级: 药物/器械 > 疾病 > 医院
        - 匹配质量: 精确匹配 > 模糊匹配
        - 置信度: 高置信度 > 低置信度
        
        应用场景:
        输入: {"白癜风":..., "308激光治疗仪":...}
        输出: "308激光治疗仪" (器械优先级高于疾病)
        
        Args:
            matched_terms: KeywordMatcher返回的匹配结果字典
        Returns:
            Optional[str]: 最关键的术语名称，无匹配时返回None
        """
        if not matched_terms:
            return None
            
        # 按优先级和置信度排序 
        sorted_terms = self._sort_terms_by_priority(matched_terms)
        
        if sorted_terms:
            # sorted_terms[0]：("308激光治疗仪", {"type": "device", "priority": 100, "confidence": 95})
            # sorted_terms[0][0]："308激光治疗仪"
            return sorted_terms[0][0]  # 返回最高优先级的术语
        
        return None
    
    def _sort_terms_by_priority(self, matched_terms: Dict) -> List[Tuple[str, Dict]]:
        """
        按优先级和置信度对术语进行排序 - 多维度综合排序算法
        
        目的: 实现复杂的术语排序逻辑，确保选择最合适的关键术语
        处理逻辑:
        1. 定义综合评分函数：业务优先级 + 置信度 + 匹配类型加分
        2. 为每个术语计算综合得分
        3. 按得分降序排列术语
        4. 返回排序后的术语列表
        
        使用技术:
        - 多维度评分: 综合考虑类型、置信度、匹配质量
        - sorted()函数: Python内置排序，稳定且高效
        - lambda表达式: 灵活定义排序规则
        - 加权计算: 不同维度贡献不同权重
        
        评分公式:
        总分 = 类型优先级 + 置信度/100 + 匹配类型加分
        - 类型优先级: 25-100分（基础分）
        - 置信度贡献: 0-1分（精细调节）  
        - 精确匹配加分: +10分（质量奖励）
        
        排序示例:
        输入: [("308激光治疗仪", 精确, 器械), ("白癜风", 精确, 疾病)]
        计算: [110分, 60分] 
        输出: 308激光治疗仪排在第一位
        
        Args:
            matched_terms: 术语匹配字典
        Returns:
            List[Tuple[str, Dict]]: 排序后的(术语名, 信息字典)列表
        """
        def sort_key(item):
            term, info = item
            # 先按术语类型优先级排序，再按置信度排序
            type_priority = info.get('priority', 0)
            confidence = info.get('confidence', 0)
            
            # 精确匹配的额外加权
            match_type_bonus = 10 if info.get('match_type') == 'exact' else 0
            
            # 综合评分：类型优先级 + 置信度/100 + 匹配类型加分
            total_score = type_priority + confidence/100 + match_type_bonus
            
            return total_score
        
        # 按评分降序排序

        #  matched_terms.items() 返回 (key, value) 元组的迭代器
        # 对于每个元组，sorted() 自动调用 sort_key(元组)
        # 所以 item 参数实际上接收的是每个 (术语名, 信息字典) 元组
        sorted_items = sorted(matched_terms.items(), key=sort_key, reverse=True)
        return sorted_items
    
    def get_term_explanation(self, term: str, term_info: Dict) -> str:
        """
        获取术语的解释信息（用于调试）
        Args:
            term: 术语
            term_info: 术语信息字典
        Returns:
            术语解释字符串
        """
        term_type = term_info.get('type', 'unknown')
        confidence = term_info.get('confidence', 0)
        match_type = term_info.get('match_type', 'unknown')
        
        type_names = {
            'drug': '药物',
            'device': '医疗器械', 
            'disease': '疾病',
            'hospital': '医院'
        }
        
        type_name = type_names.get(term_type, '未知类型')
        match_name = '精确匹配' if match_type == 'exact' else '模糊匹配'
        
        return f"{term}({type_name}, {match_name}, 置信度:{confidence})"
    
    def extract_all_relevant_terms(self, matched_terms: Dict, max_count: int = 3) -> List[Tuple[str, Dict]]:
        """
        提取所有相关术语（按优先级排序）
        Args:
            matched_terms: 匹配结果字典
            max_count: 最多返回的术语数量
        Returns:
            相关术语列表，格式为[(术语, 信息字典), ...]
        """
        sorted_terms = self._sort_terms_by_priority(matched_terms)
        return sorted_terms[:max_count]
    
    def analyze_term_distribution(self, matched_terms: Dict) -> Dict[str, int]:
        """
        分析术语类型分布
        Args:
            matched_terms: 匹配结果字典
        Returns:
            术语类型统计字典
        """
        distribution = {
            'drug': 0,
            'device': 0, 
            'disease': 0,
            'hospital': 0
        }
        
        for term, info in matched_terms.items():
            term_type = info.get('type', 'unknown')
            if term_type in distribution:
                distribution[term_type] += 1
        
        return distribution
    
    def has_high_priority_terms(self, matched_terms: Dict) -> bool:
        """
        检查是否包含高优先级术语（药物或医疗器械）
        Args:
            matched_terms: 匹配结果字典
        Returns:
            是否包含高优先级术语
        """
        for term, info in matched_terms.items():
            term_type = info.get('type', 'unknown')
            if term_type in ['drug', 'device']:
                return True
        return False
    
    def get_priority_explanation(self) -> str:
        """
        获取优先级规则说明
        Returns:
            优先级规则说明文本
        """
        return """
术语优先级规则：
1. 药物和医疗器械：最高优先级（100分）
2. 疾病：中等优先级（50分）  
3. 医院：较低优先级（25分）

在同等优先级下，精确匹配优于模糊匹配，置信度高的优于置信度低的。
        """