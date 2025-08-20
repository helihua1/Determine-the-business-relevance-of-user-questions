#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗问句相关度分析主程序
根据业务需求对用户提问进行相关度评分和关键术语提取

输出格式：[相关度分数][句子中的最关键的术语][提问的中文句子]
术语优先级：药物和医疗器械 > 疾病 > 医院
"""

import os
import sys

import pandas as pd
import yaml
# import pandas as pd  # 暂时移除，项目中未使用
from typing import List, Dict, Tuple
from tqdm import tqdm

# 导入自定义模块
# （术语加载器）
from term_loader import TermLoader
# （关键词匹配器）
from keyword_matcher import KeywordMatcher
# （语义分析器）
from semantic_analyzer import SemanticAnalyzer
# （术语提取器）
from term_extractor import TermExtractor
# # （Hugging Face配置）
# import hf_config

class MedicalQuestionAnalyzer:
    """医疗问句分析器主类"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化分析器
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        
        # 初始化各个组件，TermLoader() 是类的实例化
        self.term_loader = TermLoader(self.config)
        self.keyword_matcher = KeywordMatcher(self.config)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.term_extractor = TermExtractor()
        
        # self 参数的特殊性：
        # self 是实例方法的第一个参数：在 Python 中，当你定义一个类的方法时，第一个参数通常命名为 self（虽然可以用其他名字，但约定俗成用 self）
        # 调用时不需要显式传递 self：当你通过实例调用方法时，Python 会自动将实例对象作为第一个参数传递给 self
            #         # 定义方法时
            # def _initialize_components(self):  # self 参数
            #     # 方法体
            #     pass

            # # 调用方法时
            # self._initialize_components()  # 不需要传递参数
        self._initialize_components()
        
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("配置文件加载成功")
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
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
        """初始化各组件"""
        print("正在初始化分析器组件...")
        
        # # 设置Hugging Face离线模式，避免网络连接问题
        # try:
        #     hf_config.setup_hf_offline_mode()
        # except Exception as e:
        #     print(f"设置离线模式失败: {e}")
        
        # 加载术语库
        drugs, devices, diseases, hospitals, samples = self.term_loader.load_terms_from_file()
        
        # 初始化关键词匹配器（不包含samples）
        self.keyword_matcher.load_terms(drugs, devices, diseases, hospitals)
        
        # 初始化语义分析器（包含samples）
        try:
            print("正在初始化语义分析器...")
            self.semantic_analyzer.build_medical_corpus(drugs, devices, diseases, hospitals, samples)
            self.semantic_analyzer.load_model()
            self.semantic_analyzer.encode_medical_terms()
            print("语义分析器初始化成功！")
        except Exception as e:
            print(f"语义分析器初始化失败！！！！！！！！！！！！！！！！！！！！！！！！！！！！: {e}")
            print("将只使用关键词匹配功能")
            self.semantic_analyzer = None
            
            # # 尝试重新初始化一次
            # try:
            #     print("尝试重新初始化语义分析器...")
            #     self.semantic_analyzer = SemanticAnalyzer(self.config)
            #     self.semantic_analyzer.build_medical_corpus(drugs, devices, diseases, hospitals)
            #     self.semantic_analyzer.load_model()
            #     self.semantic_analyzer.encode_medical_terms()
            #     print("语义分析器重新初始化成功！")
            # except Exception as retry_error:
            #     print(f"重新初始化也失败: {retry_error}")
            #     print("最终将只使用关键词匹配功能")
            #     self.semantic_analyzer = None
        
        print("组件初始化完成！")
    
    def analyze_single_sentence(self, sentence: str) -> Tuple[float, str, str]:
        """
        分析单个句子
        Args:
            sentence: 输入句子
        Returns:
            (相关度分数, 关键术语, 原句子)
        """
        if not sentence or not sentence.strip():
            return 0.0, "", sentence
        
        sentence = sentence.strip()
        # # 流程一
        # # 第一步：关键词匹配 进行精确匹配和模糊匹配
        #
        # # 返回数据结构 matched_terms:
        # # {
        # #     'term1': {
        # #         'confidence': int,      # 置信度 0-100
        # #         'match_type': str,      # 'exact' 或 'fuzzy'
        # #         'priority': int,        # 业务优先级分数
        # #         'type': str            # 'drug'/'device'/'disease'/'hospital'
        # #     }
        # # }
        #
        # matched_terms = self.keyword_matcher.find_all_matches(sentence)
        #
        #
        # # 流程二
        # # 第二步：提取最关键的术语
        # key_term = self.term_extractor.extract_key_term(matched_terms)
        # if not key_term:
        #     key_term = ""
        #
        # # 流程三
        # # 第三步：计算相关度分数
        # relevance_score = self._calculate_relevance_score(sentence, matched_terms)

        relevance_score, key_term = self._calculate_relevance_score(sentence)
        # key_term = "只运行语义分析"

        return relevance_score, key_term, sentence

    def _calculate_relevance_score(self, sentence: str) -> Tuple[float, str]:
        """不运行模糊和精确匹配，只运行RAG判断的方法"""

        if self.semantic_analyzer is not None:
            try:
                semantic_score, term = self.semantic_analyzer.calculate_similarity(sentence)
                return semantic_score, term
            except Exception as e:
                print(f"语义分析出错: {e}")
                return 0.10
        return 0.00
    #
    # def _calculate_relevance_score(self, sentence: str, matched_terms: Dict) -> float:
    #     """
    #     计算相关度分数
    #     Args:
    #         sentence: 输入句子
    #         matched_terms: 匹配的术语信息
    #     Returns:
    #         相关度分数 (0.0-1.0)
    #     """
    #     # 如果有精确匹配的高优先级术语，给高分
    #     if self._has_exact_high_priority_match(matched_terms):
    #         return 0.95
    #
    #     # 如果有模糊匹配的高优先级术语，给较高分
    #     if self._has_fuzzy_high_priority_match(matched_terms):
    #         return 0.85
    #
    #     # 如果有疾病相关术语，给中等分数
    #     if self._has_disease_match(matched_terms):
    #         return 0.70
    #
    #     # 如果有医院相关术语，给较低分数
    #     if self._has_hospital_match(matched_terms):
    #         return 0.60
    #
    #     # 如果没有关键词匹配，使用语义分析
    #     if self.semantic_analyzer is not None:
    #         try:
    #             semantic_score = self.semantic_analyzer.calculate_similarity(sentence)
    #             # 将语义相似度映射到合适的范围
    #             if semantic_score > 0.8:
    #                 return 0.75  # 高语义相似度
    #             elif semantic_score > 0.6:
    #                 return 0.55  # 中等语义相似度
    #             elif semantic_score > 0.4:
    #                 return 0.35  # 较低语义相似度
    #             else:
    #                 return 0.15  # 很低相关度
    #         except Exception as e:
    #             print(f"语义分析出错: {e}")
    #             return 0.10
    #
    #     # 如果没有任何匹配，返回很低的相关度
    #     return 0.10
    #
    #
    #
    #
    # def _has_exact_high_priority_match(self, matched_terms: Dict) -> bool:
    #     """检查是否有精确匹配的高优先级术语（药物或医疗器械）"""
    #     for term, info in matched_terms.items():
    #         if (info.get('match_type') == 'exact' and
    #             info.get('type') in ['drug', 'device']):
    #             return True
    #     return False
    #
    # def _has_fuzzy_high_priority_match(self, matched_terms: Dict) -> bool:
    #     """检查是否有模糊匹配的高优先级术语（药物或医疗器械）"""
    #     for term, info in matched_terms.items():
    #         if (info.get('match_type') == 'fuzzy' and
    #             info.get('type') in ['drug', 'device'] and
    #             info.get('confidence', 0) >= 70):
    #             return True
    #     return False
    #
    # def _has_disease_match(self, matched_terms: Dict) -> bool:
    #     """检查是否有疾病相关术语匹配"""
    #     for term, info in matched_terms.items():
    #         if info.get('type') == 'disease':
    #             return True
    #     return False
    #
    # def _has_hospital_match(self, matched_terms: Dict) -> bool:
    #     """检查是否有医院相关术语匹配"""
    #     for term, info in matched_terms.items():
    #         if info.get('type') == 'hospital':
    #             return True
    #     return False
    
    def analyze_batch(self, sentences: List[str]) -> List[Tuple[float, str, str]]:
        """
        批量分析句子
        Args:
            sentences: 句子列表
        Returns:
            分析结果列表，格式为[(相关度分数, 关键术语, 原句子), ...]
        """
        results = []
        
        # 判断是否显示进度条
        show_progress = self.config['batch_processing']['show_progress']
        
        if show_progress:
            # 使用tqdm显示进度
            iterator = tqdm(sentences, desc="分析句子", unit="句")
        else:
            iterator = sentences

            # 遍历分析单个句子
        for sentence in iterator:
            try:
                result = self.analyze_single_sentence(sentence)
                results.append(result)
            except Exception as e:
                print(f"分析句子时出错: '{sentence}' - {e}")
                # 出错时返回默认值
                results.append((0.0, "", sentence))
        
        return results
    
    def format_output(self, results: List[Tuple[float, str, str]]) -> List[str]:
        """
        格式化输出结果
        Args:
            results: 分析结果列表
        Returns:
            格式化后的字符串列表
        """
        formatted_results = []
        decimal_places = self.config['output']['decimal_places']
        
        for score, key_term, sentence in results:
            # 格式化相关度分数
            score_str = f"{score:.{decimal_places}f}"
            
            # 组装输出格式：[相关度分数][关键术语][原句子]
            formatted_line = f"[{score_str}][{key_term}][{sentence}]"
            formatted_results.append(formatted_line)
        
        return formatted_results
    
    def save_results(self, results: List[str], output_file: str):
        """
        保存结果到文件
        Args:
            results: 格式化后的结果列表
            output_file: 输出文件路径
        """
        try:
            encoding = self.config['output']['encoding']
            with open(output_file, 'w', encoding=encoding) as f:
                for line in results:
                    f.write(line + '\n')
            print(f"结果已保存到: {output_file}")


        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def process_file(self, input_file: str, output_file: str = None):
        """
        开始处理文件,处理文件中的句子
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）
        """
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return
        
        if output_file is None:
            # 自动生成输出文件名
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_analyzed.txt"
        
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            print(f"从文件读取了 {len(sentences)} 个句子")
            
            # 批量分析
            results = self.analyze_batch(sentences)
            
            # 格式化输出
            formatted_results = self.format_output(results)
            
            # 保存结果
            self.save_results(formatted_results, output_file)

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
def main():
    """主函数"""
    print("=== 医疗问句相关度分析系统 ===")
    print("正在初始化系统...")
    
    # 创建分析器实例
    try:
        analyzer = MedicalQuestionAnalyzer()
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查依赖包是否正确安装，可以运行 python install_dependencies.py")
        return
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"处理文件: {input_file}")
        analyzer.process_file(input_file, output_file)
    else:
        # 交互式模式
        print("\n系统初始化完成！")
        print("使用说明：")
        # print("1. 输入句子进行单句分析")
        print("1. 输入 file '文件路径' 处理文件")
        print("3. 输入 'quit' 或 'exit' 退出")
        
        while True:
            try:
                user_input = input("\n请输入句子或命令: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                # 输入file+路径
                # if user_input.lower().startswith('file '):
                #     file_path = user_input[5:].strip()
                #
                #     # 开始处理文件
                #     analyzer.process_file(file_path)
                #     continue

                #直接读取输入为路径
                analyzer.process_file(user_input.lower())

                if not user_input:
                    continue
                
                # # 单句分析
                # score, key_term, sentence = analyzer.analyze_single_sentence(user_input)
                # result = analyzer.format_output([(score, key_term, sentence)])
                # print(f"分析结果: {result[0]}")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"处理时出错: {e}")

 

if __name__ == "__main__":


    print(f"Python可执行文件路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    main()