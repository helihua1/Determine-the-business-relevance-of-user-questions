#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义空间可视化工具
使用PCA/t-SNE降维和Plotly进行交互式3D可视化
支持多类别术语可视化
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import torch


class SemanticVisualizer:
    """语义空间可视化类"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self, model_path=None):
        """加载模型"""
        if model_path:
            self.model = SentenceTransformer(model_path)
        else:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        if self.device.type == 'cuda':
            self.model = self.model.to(self.device)

    def get_embeddings(self, terms: List[str], batch_size=32) -> np.ndarray:
        """获取术语的嵌入向量"""
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(
            terms,
            batch_size=batch_size,
            device=self.device,
            show_progress_bar=True
        )
        return embeddings

    def reduce_dimensions(self, embeddings: np.ndarray, method='pca', n_components=3) -> np.ndarray:
        """降维到3D空间"""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        return reducer.fit_transform(embeddings)

    def create_multi_category_visualization(self,
                                            category_terms: Dict[str, List[str]],
                                            embeddings: np.ndarray,
                                            title: str = "语义空间分布",
                                            method: str = 'pca') -> go.Figure:
        """创建多类别3D可视化图"""

        # 降维
        coords_3d = self.reduce_dimensions(embeddings, method=method)

        # 创建DataFrame
        data = []
        term_list = []
        label_list = []

        for category, terms in category_terms.items():
            term_list.extend(terms)
            label_list.extend([category] * len(terms))

        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2] if coords_3d.shape[1] > 2 else np.zeros(len(coords_3d)),
            'term': term_list,
            'category': label_list
        })

        # 创建3D散点图
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='category',
            hover_data=['term'],
            title=f"{title} - {method.upper()}降维",
            labels={'category': '类别'}
        )

        # 更新布局
        fig.update_layout(
            scene=dict(
                xaxis_title='维度1',
                yaxis_title='维度2',
                zaxis_title='维度3'
            ),
            width=1000,
            height=800
        )

        return fig

    def compare_models_multi_category(self,
                                      category_terms: Dict[str, List[str]],
                                      model1_embeddings: np.ndarray,
                                      model2_embeddings: np.ndarray,
                                      model1_name: str = "训练前",
                                      model2_name: str = "训练后",
                                      method: str = 'pca') -> go.Figure:
        """比较两个模型的多类别语义空间分布"""

        # 降维
        coords_3d_1 = self.reduce_dimensions(model1_embeddings, method=method)
        coords_3d_2 = self.reduce_dimensions(model2_embeddings, method=method)

        # 准备数据
        term_list = []
        category_list = []

        for category, terms in category_terms.items():
            term_list.extend(terms)
            category_list.extend([category] * len(terms))

        # 创建DataFrame
        df1 = pd.DataFrame({
            'x': coords_3d_1[:, 0],
            'y': coords_3d_1[:, 1],
            'z': coords_3d_1[:, 2] if coords_3d_1.shape[1] > 2 else np.zeros(len(coords_3d_1)),
            'term': term_list,
            'category': category_list,
            'model': model1_name
        })

        df2 = pd.DataFrame({
            'x': coords_3d_2[:, 0],
            'y': coords_3d_2[:, 1],
            'z': coords_3d_2[:, 2] if coords_3d_2.shape[1] > 2 else np.zeros(len(coords_3d_2)),
            'term': term_list,
            'category': category_list,
            'model': model2_name
        })

        df = pd.concat([df1, df2])

        # 创建3D散点图
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='category',
            symbol='model',
            hover_data=['term'],
            title=f"模型比较 - {method.upper()}降维",
            labels={'category': '类别', 'model': '模型'}
        )

        # 更新布局
        fig.update_layout(
            scene=dict(
                xaxis_title='维度1',
                yaxis_title='维度2',
                zaxis_title='维度3'
            ),
            width=1200,
            height=800
        )

        return fig


def visualize_medical_terms_multi_category(analyzer,
                                           category_terms: Dict[str, List[str]],
                                           save_path: str = "visualization.html") -> go.Figure:
    """
    可视化多类别医疗术语在语义空间中的分布

    Args:
        analyzer: SemanticAnalyzer实例
        category_terms: 字典，键为类别名称，值为术语列表
        save_path: 保存路径

    Returns:
        plotly图形对象
    """
    # 合并所有术语
    all_terms = []
    for terms in category_terms.values():
        all_terms.extend(terms)

    # 获取当前模型的嵌入
    print("获取当前模型嵌入...")
    current_embeddings = analyzer.model.encode(all_terms, device=analyzer.device)

    # 创建可视化器
    visualizer = SemanticVisualizer(device=analyzer.device)

    # 获取预训练模型的嵌入（用于比较）
    print("获取预训练模型嵌入...")
    visualizer.load_model()  # 加载原始预训练模型
    pretrained_embeddings = visualizer.get_embeddings(all_terms)

    # 创建比较可视化
    print("创建可视化...")
    fig = visualizer.compare_models_multi_category(
        category_terms, pretrained_embeddings, current_embeddings,
        "预训练模型", "微调后模型", method='pca'
    )

    # 保存为HTML文件（可交互）
    if save_path:
        fig.write_html(save_path)
        print(f"可视化已保存到: {save_path}")
    fig.show()
    return fig


# 向后兼容的旧函数
def visualize_medical_terms(analyzer,
                            baidianfeng_terms: List[str],
                            other_terms: List[str],
                            A_terms: Optional[List[str]] = None,
                            B_terms: Optional[List[str]] = None,
                            C_terms: Optional[List[str]] = None,
                            save_path: str = "visualization.html") -> go.Figure:
    """
    可视化医疗术语在语义空间中的分布（支持多类别）

    Args:
        analyzer: SemanticAnalyzer实例
        baidianfeng_terms: 白癜风相关术语列表
        other_terms: 其他疾病术语列表
        A_terms: A类术语列表
        B_terms: B类术语列表
        C_terms: C类术语列表
        save_path: 保存路径

    Returns:
        plotly图形对象
    """
    # 构建类别字典
    category_terms = {
        '白癜风相关': baidianfeng_terms,
        '其他疾病': other_terms
    }

    # 添加额外的类别
    if A_terms:
        category_terms['A类术语'] = A_terms
    if B_terms:
        category_terms['B类术语'] = B_terms
    if C_terms:
        category_terms['C类术语'] = C_terms

    return visualize_medical_terms_multi_category(analyzer, category_terms, save_path)