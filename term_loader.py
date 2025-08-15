#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
术语加载器
从术语库文件中加载各类医疗术语
"""
from typing import Dict, List, Tuple
import os

class TermLoader:
    """术语加载器类"""
    
    def __init__(self, terms_file: str = "medical_terms.txt"):
        """
        初始化术语加载器 - 创建医疗术语文件解析器
        
        目的: 初始化术语文件加载器，指定数据源文件
        处理逻辑:
        1. 保存术语文件路径配置
        2. 设定默认文件为medical_terms.txt
        3. 为后续文件解析做准备
        
        使用技术:
        - 配置管理: 灵活指定数据源文件
        - 默认值设计: 简化常用场景的调用
        
        设计模式:
        - 单一职责: 专门负责术语数据的加载和解析
        - 依赖注入: 通过参数灵活指定数据源
        
        Args:
            terms_file: 医疗术语库文件路径，默认为"medical_terms.txt"
        """
        self.terms_file = terms_file
        
    def load_terms_from_file(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        从文件加载术语 - 解析医疗术语文件并按类别分类
        
        目的: 从structured文本文件中提取四类医疗术语
        处理逻辑:
        1. 检查术语文件是否存在，不存在则使用默认术语
        2. 按行读取文件，忽略注释行(#)和空行
        3. 识别分类标记([DRUG], [DEVICE], [DISEASE], [HOSPITAL])
        4. 将术语分配到对应的类别列表
        5. 处理文件读取异常，自动回退到默认术语
        6. 输出加载统计信息
        
        使用技术:
        - 文件I/O: 安全的文件读取操作
        - 文本解析: 逐行处理，状态机模式
        - 异常处理: 文件操作的错误恢复
        - 分类标记解析: [TAG]格式的标签识别
        - 编码处理: UTF-8确保中文正确读取
        
        文件格式规范:
        ```
        # 注释行，会被忽略
        [DRUG]          # 药物分类开始
        雷公藤
        他克莫司
        [DEVICE]        # 医疗器械分类开始  
        308激光治疗仪
        光疗仪
        ```
        
        错误处理:
        - 文件不存在: 自动使用默认术语库
        - 读取错误: 异常捕获，回退到默认术语  
        - 格式错误: 忽略无效行，继续处理
        - 未知分类: 警告提示，跳过该术语
        
        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: 
            (药物术语列表, 医疗器械术语列表, 疾病术语列表, 医院术语列表)
        """
        drugs = []
        devices = []
        diseases = []
        hospitals = []
        
        if not os.path.exists(self.terms_file):
            print(f"警告：术语文件 {self.terms_file} 不存在，使用默认术语")
            return self._get_default_terms()
        
        try:
            current_category = None
            
            # 文件对象 f 的特征：
            # 类型：Python 内置的 file 对象
            # 打开模式：'r' - 只读模式
            with open(self.terms_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    
                    # 检查分类标记
                    if line.startswith('[') and line.endswith(']'):
                        # [1:-1] 表示从索引1开始到倒数第二个字符结束（不包含最后一个字符）目的是去掉方括号 [ 和 ]
                        category = line[1:-1].upper()#
                        if category in ['DRUG', 'DEVICE', 'DISEASE', 'HOSPITAL']:
                            current_category = category
                        else:
                            print(f"警告：第{line_num}行包含未知分类标记: {line}")
                        continue
                    
                    # 添加术语到对应分类
                    if current_category == 'DRUG':
                        drugs.append(line)
                    elif current_category == 'DEVICE':
                        devices.append(line)
                    elif current_category == 'DISEASE':
                        diseases.append(line)
                    elif current_category == 'HOSPITAL':
                        hospitals.append(line)
                    else:
                        print(f"警告：第{line_num}行的术语'{line}'没有指定分类，将被忽略")
            
            print(f"术语加载完成：药物{len(drugs)}个，医疗器械{len(devices)}个，"
                  f"疾病{len(diseases)}个，医院{len(hospitals)}个")
            
            return drugs, devices, diseases, hospitals
            
        except Exception as e:
            print(f"加载术语文件时出错: {e}")
            print("使用默认术语")
            return self._get_default_terms()
    
    def _get_default_terms(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        获取默认术语 - 文件加载失败时的备用术语库
        
        目的: 提供硬编码的默认医疗术语，确保系统在文件缺失时仍能运行
        处理逻辑:
        1. 定义核心药物术语列表
        2. 定义核心医疗器械术语列表
        3. 定义核心疾病术语列表  
        4. 定义核心医院术语列表
        5. 返回四个分类的术语元组
        
        使用技术:
        - 硬编码数据: 确保基本功能可用
        - 精选术语: 包含最常见和重要的医疗术语
        - 分类完整: 覆盖四个主要医疗术语类别
        
        备用术语选择原则:
        - 高频使用: 选择常见的医疗术语
        - 业务相关: 与白癜风、银屑病业务密切相关
        - 覆盖完整: 每个类别都有代表性术语
        
        应用场景:
        - medical_terms.txt文件不存在
        - 文件读取权限不足
        - 文件格式错误导致解析失败
        - 系统初始化时的应急方案
        
        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: 
            默认的(药物, 器械, 疾病, 医院)术语列表
        """
        drugs = [
            "雷公藤", "他克莫司", "卡泊三醇", "甲氨蝶呤", "环孢素",
            "白癜风丸", "白灵片", "驱白巴布期片"
        ]
        
        devices = [
            "308激光治疗仪", "308准分子激光", "311窄谱UVB", 
            "UVB紫外线光疗仪", "光疗仪", "激光治疗仪"
        ]
        
        diseases = [
            "白癜风", "银屑病", "牛皮癣", "局限性白癜风", "泛发性白癜风",
            "寻常型银屑病", "皮肤病", "白斑病"
        ]
        
        hospitals = [
            "宁波鄞州博润皮肤病医院", "博润皮肤病医院", "皮肤病医院",
            "皮肤科医院", "白癜风医院", "银屑病医院"
        ]
        
        return drugs, devices, diseases, hospitals
    
    def add_terms_to_file(self, new_drugs: List[str] = None, 
                         new_devices: List[str] = None,
                         new_diseases: List[str] = None, 
                         new_hospitals: List[str] = None):
        """
        向术语文件添加新术语
        Args:
            new_drugs: 新增药物术语列表
            new_devices: 新增医疗器械术语列表  
            new_diseases: 新增疾病术语列表
            new_hospitals: 新增医院术语列表
        """
        if not any([new_drugs, new_devices, new_diseases, new_hospitals]):
            return
        
        try:
            with open(self.terms_file, 'a', encoding='utf-8') as f:
                f.write('\n# === 新增术语 ===\n')
                
                if new_drugs:
                    f.write('\n[DRUG]\n')
                    for drug in new_drugs:
                        f.write(f'{drug}\n')
                
                if new_devices:
                    f.write('\n[DEVICE]\n')
                    for device in new_devices:
                        f.write(f'{device}\n')
                
                if new_diseases:
                    f.write('\n[DISEASE]\n') 
                    for disease in new_diseases:
                        f.write(f'{disease}\n')
                
                if new_hospitals:
                    f.write('\n[HOSPITAL]\n')
                    for hospital in new_hospitals:
                        f.write(f'{hospital}\n')
            
            print("新术语已添加到文件")
            
        except Exception as e:
            print(f"添加术语到文件时出错: {e}")
    
    def get_term_statistics(self) -> Dict[str, int]:
        """
        统计每个术语类别的数量
        Returns:
            术语统计字典
        """
        #  等价于  
        #   result = self.load_terms_from_file()
        #   drugs = result[0]
        #   devices = result[1]
        #   diseases = result[2]
        #   hospitals = result[3]
        drugs, devices, diseases, hospitals = self.load_terms_from_file()
        
        return {
            'drugs': len(drugs),
            'devices': len(devices), 
            'diseases': len(diseases),
            'hospitals': len(hospitals),
            'total': len(drugs) + len(devices) + len(diseases) + len(hospitals)
        }