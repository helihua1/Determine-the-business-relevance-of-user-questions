#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义空间可视化工具
使用PCA/t-SNE降维和Plotly进行交互式3D可视化
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Tuple
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

    def create_visualization(self,
                             terms: List[str],
                             labels: List[str],
                             embeddings: np.ndarray,
                             title: str = "语义空间分布",
                             method: str = 'pca') -> go.Figure:
        """创建3D可视化图"""

        # 降维
        coords_3d = self.reduce_dimensions(embeddings, method=method)

        # 创建DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2] if coords_3d.shape[1] > 2 else np.zeros(len(coords_3d)),
            'term': terms,
            'label': labels
        })

        # 创建3D散点图
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            hover_data=['term'],
            title=f"{title} - {method.upper()}降维",
            labels={'label': '类别'}
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

    def compare_models(self,
                       terms: List[str],
                       labels: List[str],
                       model1_embeddings: np.ndarray,
                       model2_embeddings: np.ndarray,
                       model1_name: str = "训练前",
                       model2_name: str = "训练后",
                       method: str = 'pca') -> go.Figure:
        """比较两个模型的语义空间分布"""

        # 降维
        coords_3d_1 = self.reduce_dimensions(model1_embeddings, method=method)
        coords_3d_2 = self.reduce_dimensions(model2_embeddings, method=method)

        # 创建DataFrame
        df1 = pd.DataFrame({
            'x': coords_3d_1[:, 0],
            'y': coords_3d_1[:, 1],
            'z': coords_3d_1[:, 2] if coords_3d_1.shape[1] > 2 else np.zeros(len(coords_3d_1)),
            'term': terms,
            'label': labels,
            'model': model1_name
        })

        df2 = pd.DataFrame({
            'x': coords_3d_2[:, 0],
            'y': coords_3d_2[:, 1],
            'z': coords_3d_2[:, 2] if coords_3d_2.shape[1] > 2 else np.zeros(len(coords_3d_2)),
            'term': terms,
            'label': labels,
            'model': model2_name
        })

        df = pd.concat([df1, df2])

        # 创建3D散点图
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            symbol='model',
            hover_data=['term'],
            title=f"模型比较 - {method.upper()}降维",
            labels={'label': '类别', 'model': '模型'}
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


def visualize_medical_terms(analyzer,
                            baidianfeng_terms: List[str],
                            other_terms: List[str],
                            save_path: str = "visualization.html"):
    """
    可视化医疗术语在语义空间中的分布
    """
    # 合并所有术语
    all_terms = baidianfeng_terms + other_terms
    labels = ['白癜风相关'] * len(baidianfeng_terms) + ['其他疾病'] * len(other_terms)

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
    fig = visualizer.compare_models(
        all_terms, labels, pretrained_embeddings, current_embeddings,
        "预训练模型", "微调后模型", method='pca'
    )

    # 保存为HTML文件（可交互）
    fig.write_html(save_path)
    print(f"可视化已保存到: {save_path}")

    # 也可以显示在浏览器中
    fig.show()

    return fig