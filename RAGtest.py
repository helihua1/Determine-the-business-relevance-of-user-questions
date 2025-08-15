from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 初始化Embedding模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 准备数据并生成向量
documents = ["糖尿病治疗药物", "高血压饮食建议", "冠心病预防措施"]
doc_vectors = model.encode(documents)

# 3. 构建RAG数据库（FAISS）

#doc_vectors.shape[0] == 3（文本数量）
# doc_vectors.shape[1] == 384（每个向量的维度）
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  #  初始化索引时必须指定维度
index.add(doc_vectors)

# 4. 检索示例
query = "怎么降血糖？"
query_vector = model.encode([query])
D, I = index.search(query_vector, k=2)  # 返回前2个结果

print("最相关文档：", [documents[i] for i in I[0]])