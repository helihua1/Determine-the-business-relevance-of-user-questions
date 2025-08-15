#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版医疗问句相关度分析器
暂时不使用语义分析功能，仅基于关键词匹配
"""

import os
import sys
import yaml
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

# 导入自定义模块
from term_loader import TermLoader
from keyword_matcher import KeywordMatcher
from term_extractor import TermExtractor

class SimpleMedicalAnalyzer:
    """简化版医疗问句分析器"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        【流程1】系统初始化 - 创建分析器实例并加载配置
        功能: 读取config.yaml配置文件，初始化各个处理模块
        """
        self.config = self._load_config(config_file)
        
        # 初始化组件
        self.term_loader = TermLoader()
        self.keyword_matcher = KeywordMatcher(self.config)
        self.term_extractor = TermExtractor()
        
        # 加载术语数据
        self._initialize_components()
        
    def _load_config(self, config_file: str) -> Dict:
        """
        【流程1.1】配置加载 - 读取config.yaml配置文件
        
        目的: 加载系统运行所需的所有配置参数
        处理逻辑:
        1. 尝试读取YAML配置文件
        2. 解析配置项（相似度阈值、批处理参数、输出格式等）
        3. 异常时返回默认配置确保系统正常运行
        
        使用技术:
        - yaml.safe_load(): 安全解析YAML文件，防止代码注入
        - 异常处理: try-catch确保配置加载的健壮性
        - 默认配置回退机制
        
        Args:
            config_file: 配置文件路径，默认为config.yaml
        Returns:
            Dict: 配置参数字典，包含阈值、批处理、输出等设置
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("配置文件加载成功")
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """
        获取默认配置 - 配置文件加载失败时的回退方案
        
        目的: 提供系统默认配置，确保程序能正常运行
        处理逻辑:
        1. 定义相似度阈值（高0.8、低0.6、模糊0.7）
        2. 设置模糊匹配参数（最小长度2、编辑距离阈值2）
        3. 配置批处理设置（批次大小1000、显示进度）
        4. 设置输出格式（小数位2位、UTF-8编码）
        
        使用技术:
        - 硬编码配置字典
        - 结构化配置组织
        
        Returns:
            Dict: 默认配置字典，包含所有必需的配置项
        """
        return {
            'similarity_thresholds': {
                'high_confidence': 0.8,
                'low_confidence': 0.6,
                'fuzzy_match': 0.7
            },
            'fuzzy_match': {
                'min_length': 2,
                'edit_distance_threshold': 2
            },
            'batch_processing': {
                'batch_size': 1000,
                'show_progress': True
            },
            'output': {
                'decimal_places': 2,
                'encoding': 'utf-8'
            }
        }
    
    def _initialize_components(self):
        """
        【流程1.2】组件初始化 - 加载医疗术语库到各个处理模块
        
        目的: 初始化系统核心组件，加载医疗术语数据
        处理逻辑:
        1. 调用TermLoader加载medical_terms.txt文件
        2. 按类别解析术语（药物、医疗器械、疾病、医院）
        3. 将术语数据加载到KeywordMatcher中
        4. 将术语添加到jieba分词词典提高分词准确性
        
        使用技术:
        - TermLoader: 术语文件解析器
        - KeywordMatcher: 关键词匹配引擎
        - jieba分词: 中文自然语言处理
        - 术语分类管理
        
        数据流: medical_terms.txt → TermLoader → KeywordMatcher → jieba词典
        """
        print("正在初始化分析器组件...")
        
        # 加载术语库
        drugs, devices, diseases, hospitals = self.term_loader.load_terms_from_file()
        
        # 初始化关键词匹配器
        self.keyword_matcher.load_terms(drugs, devices, diseases, hospitals)
        
        print("组件初始化完成！")
    
    def analyze_single_sentence(self, sentence: str) -> Tuple[float, str, str]:
        """
        【流程4】单句分析核心算法 - 对单个句子进行完整的相关度分析
        
        目的: 分析单个句子的医疗相关度并提取关键术语
        处理逻辑:
        1. 输入预处理：去除首尾空格，处理空句子
        2. 关键词匹配：在句子中查找医疗术语（精确+模糊匹配）
        3. 术语提取：按优先级规则选择最关键的术语
        4. 相关度计算：基于匹配结果和上下文计算分数
        5. 结果返回：(相关度分数, 关键术语, 原句子)
        
        使用技术:
        - KeywordMatcher: FuzzyWuzzy模糊匹配 + jieba分词
        - TermExtractor: 优先级排序算法
        - 相关度评分算法: 多层判断逻辑
        - 字符串处理: strip()去空格
        
        Args:
            sentence: 待分析的中文句子
        Returns:
            Tuple[float, str, str]: (相关度分数0.0-1.0, 关键术语, 原句子)
        """
        if not sentence or not sentence.strip():
            return 0.0, "", sentence
        
        sentence = sentence.strip()
        
        # 【流程4.1】关键词匹配 - 在句子中查找医疗术语（精确+模糊匹配）
        matched_terms = self.keyword_matcher.find_all_matches(sentence)
        
        # 【流程4.2】术语提取 - 按优先级规则提取最关键的术语
        key_term = self.term_extractor.extract_key_term(matched_terms)
        if not key_term:
            key_term = ""
        
        # 【流程4.3】相关度计算 - 综合匹配结果计算最终相关度分数
        relevance_score = self._calculate_simple_relevance(sentence, matched_terms)
        
        return relevance_score, key_term, sentence
    
    def _calculate_simple_relevance(self, sentence: str, matched_terms: Dict) -> float:
        """
        【流程5】相关度评分算法 - 根据匹配的术语类型和质量计算最终分数
        
        目的: 基于医疗业务优先级和匹配质量计算句子相关度
        处理逻辑:
        1. 无匹配检查：没有找到任何医疗术语返回0.10
        2. 负面过滤：包含明显无关词汇（公交、天气等）返回0.10  
        3. 高优先级精确匹配：药物/器械精确匹配返回0.95
        4. 高优先级模糊匹配：药物/器械模糊匹配且上下文相关返回0.85
        5. 疾病术语匹配：精确0.75，模糊0.65
        6. 医院术语匹配：精确0.60，模糊0.50
        7. 其他匹配：默认0.40
        
        使用技术:
        - 负面关键词列表过滤
        - 术语类型优先级排序
        - 匹配类型区分（精确vs模糊）
        - 置信度阈值判断
        - 上下文相关性检查
        
        业务规则:
        - 药物和医疗器械: 最高优先级（客户核心需求）
        - 疾病相关: 中等优先级（相关咨询）
        - 医院相关: 较低优先级（地理位置查询）
        
        Args:
            sentence: 原始句子（用于负面过滤）
            matched_terms: 匹配的术语字典
        Returns:
            float: 相关度分数，范围0.10-0.95
        """
        if not matched_terms:
            return 0.10  # 无匹配术语，很低相关度
        
        # 先检查是否包含明显无关的关键词，进行负面过滤
        negative_keywords = ['公交', '交通', '天气', '手机', '违章', '买', '购买']
        for neg_word in negative_keywords:
            if neg_word in sentence:
                return 0.10  # 包含明显无关词汇，直接判定为不相关
        
        # 检查是否有精确匹配的高优先级术语
        exact_high_priority = []
        for term, info in matched_terms.items():
            if (info.get('match_type') == 'exact' and 
                info.get('type') in ['drug', 'device']):
                exact_high_priority.append((term, info))
        
        if exact_high_priority:
            return 0.95  # 精确匹配药物/设备，最高相关度
        
        # 检查是否有高置信度模糊匹配的高优先级术语
        fuzzy_high_priority = []
        for term, info in matched_terms.items():
            if (info.get('match_type') == 'fuzzy' and 
                info.get('type') in ['drug', 'device'] and
                info.get('confidence', 0) >= 80):
                # 额外检查：确保匹配的术语与句子语境相关
                if self._is_contextually_relevant(sentence, term):
                    fuzzy_high_priority.append((term, info))
        
        if fuzzy_high_priority:
            return 0.85  # 高置信度模糊匹配药物/设备
        
        # 检查疾病术语
        disease_matches = []
        for term, info in matched_terms.items():
            if info.get('type') == 'disease':
                disease_matches.append((term, info))
        
        if disease_matches:
            # 优先考虑精确匹配
            for term, info in disease_matches:
                if info.get('match_type') == 'exact':
                    return 0.75  # 精确匹配疾病
            # 其次考虑模糊匹配
            return 0.65  # 模糊匹配疾病
        
        # 检查医院术语
        hospital_matches = []
        for term, info in matched_terms.items():
            if info.get('type') == 'hospital':
                hospital_matches.append((term, info))
        
        if hospital_matches:
            for term, info in hospital_matches:
                if info.get('match_type') == 'exact':
                    return 0.60  # 精确匹配医院
            return 0.50  # 模糊匹配医院
        
        # 有匹配但不在以上分类，给默认分数
        return 0.40
    
    def _is_contextually_relevant(self, sentence: str, term: str) -> bool:
        """
        检查术语在句子中是否上下文相关 - 防止误判的语境检查
        
        目的: 验证匹配到的医疗术语是否在医疗语境中使用
        处理逻辑:
        1. 定义医疗相关的上下文关键词列表
        2. 检查句子中是否包含医疗语境词汇
        3. 返回是否存在医疗相关上下文
        
        使用技术:
        - 医疗语境词汇库匹配
        - 字符串包含检查（in操作符）
        - 上下文相关性验证
        
        应用场景:
        - "308激光治疗" vs "308公交" - 前者有"治疗"上下文，后者无
        - "雷公藤治疗" vs "雷公藤公司" - 前者有医疗上下文
        
        Args:
            sentence: 完整句子
            term: 匹配到的术语
        Returns:
            bool: True表示在医疗语境中，False表示非医疗语境
        """
        # 医疗相关的上下文词汇
        medical_context_words = [
            '治疗', '症状', '效果', '用药', '疗效', '副作用', '价格', 
            '怎么样', '如何', '能否', '可以', '医院', '诊断', '病情',
            '皮肤', '疾病', '药物', '仪器', '激光', '光疗'
        ]
        
        # 检查句子中是否包含医疗相关词汇
        for context_word in medical_context_words:
            if context_word in sentence:
                return True
        
        return False
    
    def analyze_batch(self, sentences: List[str]) -> List[Tuple[float, str, str]]:
        """
        【流程3】批量分析控制器 - 逐句调用分析算法并显示进度
        功能: 遍历所有句子，对每句调用analyze_single_sentence进行分析
        """
        results = []
        show_progress = self.config['batch_processing']['show_progress']
        
        if show_progress:
            iterator = tqdm(sentences, desc="分析句子", unit="句")
        else:
            iterator = sentences
        
        for sentence in iterator:
            try:
                result = self.analyze_single_sentence(sentence)
                results.append(result)
            except Exception as e:
                print(f"分析句子时出错: '{sentence}' - {e}")
                results.append((0.0, "", sentence))
        
        return results
    
    def format_output(self, results: List[Tuple[float, str, str]]) -> List[str]:
        """
        【流程6】TXT格式化 - 将分析结果转换为[分数][术语][句子]格式
        功能: 格式化为传统的文本输出格式，用于txt文件保存
        """
        formatted_results = []
        decimal_places = self.config['output']['decimal_places']
        
        for score, key_term, sentence in results:
            score_str = f"{score:.{decimal_places}f}"
            formatted_line = f"[{score_str}][{key_term}][{sentence}]"
            formatted_results.append(formatted_line)
        
        return formatted_results
    
    def save_results(self, results: List[str], output_file: str):
        """
        【流程7】TXT文件输出 - 保存格式化后的文本结果到.txt文件
        功能: 将[分数][术语][句子]格式的结果写入文本文件
        """
        try:
            encoding = self.config['output']['encoding']
            with open(output_file, 'w', encoding=encoding) as f:
                for line in results:
                    f.write(line + '\n')
            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def save_results_to_excel(self, results: List[Tuple[float, str, str]], output_file: str):
        """
        【流程8】Excel文件输出 - 创建三列表格并保存到.xlsx文件
        
        目的: 将分析结果转换为结构化的Excel表格，便于数据分析
        处理逻辑:
        1. 数据转换：将元组列表转换为pandas DataFrame
        2. 表格结构：创建三列（相关度、关键术语、原句子）
        3. Excel写入：使用openpyxl引擎保存为.xlsx格式
        4. 格式优化：调整列宽、设置表头样式
        5. 异常处理：确保文件保存的健壮性
        
        使用技术:
        - pandas DataFrame: 数据结构化处理
        - pandas ExcelWriter: Excel文件写入器
        - openpyxl引擎: Excel格式支持和样式设置
        - openpyxl.styles.Font: 字体样式设置
        - 列宽自适应: column_dimensions设置
        
        Excel格式:
        - 列A: 相关度（浮点数，0.00-1.00）
        - 列B: 关键术语（字符串，可为空）
        - 列C: 原句子（字符串）
        - 表头加粗，列宽优化（12/25/50字符）
        
        Args:
            results: 分析结果列表，格式为[(相关度分数, 关键术语, 原句子), ...]
            output_file: 输出Excel文件路径（.xlsx格式）
        """
        try:
            # 创建DataFrame
            data = []
            for score, key_term, sentence in results:
                data.append({
                    '相关度': score,
                    '关键术语': key_term,
                    '原句子': sentence
                })
            
            df = pd.DataFrame(data)
            
            # 保存到Excel文件
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='分析结果', index=False)
                
                # 获取工作表对象进行格式化
                worksheet = writer.sheets['分析结果']
                
                # 调整列宽
                worksheet.column_dimensions['A'].width = 12  # 相关度列
                worksheet.column_dimensions['B'].width = 25  # 关键术语列  
                worksheet.column_dimensions['C'].width = 50  # 原句子列
                
                # 设置表头格式
                from openpyxl.styles import Font
                header_font = Font(bold=True)
                for cell in worksheet[1]:
                    cell.font = header_font
            
            print(f"Excel结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"保存Excel结果失败: {e}")
    
    def process_file(self, input_file: str, output_file: str = None):
        """
        【流程2】文件处理入口 - 读取输入文件并协调整个分析流程
        功能: 读取test_sentences.txt → 批量分析 → 输出txt和xlsx文件
        """
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_analyzed"
        else:
            base_name = os.path.splitext(output_file)[0]
            output_file = base_name
        
        try:
            # 【流程2.1】数据读取 - 从输入文件读取原始句子数据
            with open(input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            print(f"从文件读取了 {len(sentences)} 个句子")
            
            # 【流程2.2】批量分析 - 对所有句子进行相关度分析
            results = self.analyze_batch(sentences)
            
            # 【流程2.3】结果格式化 - 将分析结果格式化为[相关度][术语][句子]格式
            formatted_results = self.format_output(results)
            
            # 【流程2.4】TXT输出 - 保存传统格式的分析结果
            txt_file = f"{output_file}.txt"
            self.save_results(formatted_results, txt_file)
            
            # 【流程2.5】Excel输出 - 保存三列表格格式的分析结果
            excel_file = f"{output_file}.xlsx"
            self.save_results_to_excel(results, excel_file)
            
            # 显示统计信息
            self._print_statistics(results)
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
    
    def _print_statistics(self, results: List[Tuple[float, str, str]]):
        """打印分析统计信息"""
        if not results:
            return
        
        scores = [score for score, _, _ in results]
        high_relevance = sum(1 for score in scores if score >= 0.8)
        medium_relevance = sum(1 for score in scores if 0.5 <= score < 0.8)
        low_relevance = sum(1 for score in scores if score < 0.5)
        
        print(f"\n=== 分析统计 ===")
        print(f"总句子数: {len(results)}")
        print(f"高相关度 (≥0.8): {high_relevance} ({high_relevance/len(results)*100:.1f}%)")
        print(f"中等相关度 (0.5-0.8): {medium_relevance} ({medium_relevance/len(results)*100:.1f}%)")
        print(f"低相关度 (<0.5): {low_relevance} ({low_relevance/len(results)*100:.1f}%)")
        print(f"平均相关度: {sum(scores)/len(scores):.3f}")


def main():
    """主函数"""
    print("=== 医疗问句相关度分析系统（简化版） ===")
    print("正在初始化系统...")
    
    try:
        analyzer = SimpleMedicalAnalyzer()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"处理文件: {input_file}")
        analyzer.process_file(input_file, output_file)
    else:
        # 交互式模式
        print("\n系统初始化完成！")
        print("使用说明：")
        print("1. 输入句子进行单句分析")
        print("2. 输入文件路径处理文件")
        print("3. 输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n请输入句子或文件路径: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # 检查是否是文件路径
                if os.path.exists(user_input):
                    analyzer.process_file(user_input)
                else:
                    # 单句分析
                    score, key_term, sentence = analyzer.analyze_single_sentence(user_input)
                    result = analyzer.format_output([(score, key_term, sentence)])
                    print(f"分析结果: {result[0]}")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"处理时出错: {e}")


if __name__ == "__main__":
    main()