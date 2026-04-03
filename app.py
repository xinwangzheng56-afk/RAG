"""
RAG 系统可视化演示 - 后端服务器
使用 DeepSeek API 替代 Gemini API
"""

import os
import json
from typing import List
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 存储全局状态
rag_state = {
    "chunks": [],
    "embeddings": [],
    "retrieved_chunks": [],
    "reranked_chunks": [],
    "answer": ""
}

def split_into_chunks(doc_content: str) -> List[str]:
    """步骤 1: 分片 - 将文档分割成小块"""
    chunks = [chunk.strip() for chunk in doc_content.split("\n\n") if chunk.strip()]
    return chunks

def embed_chunk(chunk: str) -> List[float]:
    """使用本地模型生成嵌入（简化版本，使用随机向量模拟）"""
    # 实际项目中应该使用 sentence-transformers
    # 这里为了演示简化处理
    import hashlib
    hash_val = int(hashlib.md5(chunk.encode()).hexdigest(), 16)
    # 生成 768 维向量（与 text2vec 一致）
    embedding = []
    for i in range(768):
        val = ((hash_val >> (i % 64)) & 1) * 2 - 1
        val += (hash_val + i) % 100 / 100 - 0.5
        embedding.append(float(val) / 10)
    # 归一化
    norm = sum(v*v for v in embedding) ** 0.5
    return [v / norm for v in embedding]

def save_embeddings(chunks: List[str]) -> List[dict]:
    """步骤 2: 索引 - 保存嵌入向量"""
    indexed_data = []
    for i, chunk in enumerate(chunks):
        embedding = embed_chunk(chunk)
        indexed_data.append({
            "id": str(i),
            "content": chunk,
            "embedding": embedding
        })
    return indexed_data

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    return dot_product

def retrieve(query: str, indexed_data: List[dict], top_k: int = 5) -> List[dict]:
    """步骤 3: 召回 - 检索相关文档"""
    query_embedding = embed_chunk(query)
    
    # 计算相似度
    scored_data = []
    for item in indexed_data:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_data.append({**item, "score": score})
    
    # 按相似度排序
    scored_data.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_data[:top_k]

def rerank(query: str, retrieved_chunks: List[dict], top_k: int = 3) -> List[dict]:
    """步骤 4: 重排 - 使用交叉编码器重新排序（简化版本）"""
    # 简化版本：基于关键词匹配度重新排序
    query_words = set(query.lower())
    
    reranked = []
    for chunk in retrieved_chunks:
        content = chunk["content"].lower()
        # 计算关键词匹配度
        match_score = sum(1 for word in query_words if word in content) / len(query_words)
        # 结合原始相似度
        final_score = chunk["score"] * 0.5 + match_score * 0.5
        reranked.append({**chunk, "rerank_score": final_score})
    
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def generate_answer(query: str, chunks: List[dict]) -> str:
    """步骤 5: 生成 - 使用 DeepSeek API 生成答案"""
    context = "\n\n".join([chunk["content"] for chunk in chunks])
    
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题：{query}

相关片段:
{context}

请基于上述内容作答，不要编造信息。"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用 DeepSeek API 失败：{str(e)}"

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/load_doc', methods=['POST'])
def load_doc():
    """加载文档"""
    try:
        # 读取 doc.md 文件
        doc_path = os.path.join(os.path.dirname(__file__), 'doc.md')
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分片
        chunks = split_into_chunks(content)
        
        # 索引
        indexed_data = save_embeddings(chunks)
        
        # 存储状态
        rag_state["chunks"] = [{
            "id": str(i),
            "content": chunk,
            "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
        } for i, chunk in enumerate(chunks)]
        rag_state["embeddings"] = indexed_data
        rag_state["retrieved_chunks"] = []
        rag_state["reranked_chunks"] = []
        rag_state["answer"] = ""
        
        return jsonify({
            "success": True,
            "chunks_count": len(chunks),
            "message": "文档加载成功！"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """搜索和召回"""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"success": False, "error": "请输入查询内容"}), 400
    
    # 召回
    retrieved = retrieve(query, rag_state["embeddings"], top_k=5)
    rag_state["retrieved_chunks"] = [{
        "id": chunk["id"],
        "content": chunk["content"],
        "score": round(chunk["score"], 4)
    } for chunk in retrieved]
    
    # 重排
    reranked = rerank(query, retrieved, top_k=3)
    rag_state["reranked_chunks"] = [{
        "id": chunk["id"],
        "content": chunk["content"],
        "score": round(chunk["score"], 4),
        "rerank_score": round(chunk["rerank_score"], 4)
    } for chunk in reranked]
    
    return jsonify({
        "success": True,
        "retrieved_count": len(retrieved),
        "reranked_count": len(reranked)
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """生成答案"""
    data = request.json
    query = data.get("query", "")
    
    if not rag_state["reranked_chunks"]:
        return jsonify({"success": False, "error": "请先进行搜索"}), 400
    
    # 生成答案
    answer = generate_answer(query, rag_state["reranked_chunks"])
    rag_state["answer"] = answer
    
    return jsonify({
        "success": True,
        "answer": answer
    })

@app.route('/api/state', methods=['GET'])
def get_state():
    """获取当前状态"""
    return jsonify({
        "chunks": rag_state["chunks"],
        "retrieved_chunks": rag_state["retrieved_chunks"],
        "reranked_chunks": rag_state["reranked_chunks"],
        "answer": rag_state["answer"]
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """重置状态"""
    rag_state["chunks"] = []
    rag_state["embeddings"] = []
    rag_state["retrieved_chunks"] = []
    rag_state["reranked_chunks"] = []
    rag_state["answer"] = ""
    
    return jsonify({"success": True})

if __name__ == '__main__':
    print("=" * 50)
    print("RAG 系统可视化演示服务器启动中...")
    print("=" * 50)
    print("\n请在浏览器中访问：http://localhost:5001")
    print("\n确保已设置 DEEPSEEK_API_KEY 环境变量")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5001)
