"""
RAG 分片对比实验演示系统
可视化展示不同 chunk_size 对检索效果的影响
"""
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# 全局缓存 embedding 模型
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    return embedding_model

# 默认文档
DEFAULT_TEXT = """
合同应当遵循公平原则，确定各方的权利和义务。
当事人依法享有自愿订立合同的权利，任何单位和个人不得非法干预。
当事人应当遵循诚实信用原则，履行义务。
"""

@app.route('/')
def index():
    return render_template('experiment.html')

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """运行单组实验"""
    data = request.json
    text = data.get('text', DEFAULT_TEXT)
    query = data.get('query', '合同订立应遵循哪些原则？')
    chunk_size = data.get('chunk_size', 80)

    try:
        # 1. 分片
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=2
        )
        docs = splitter.create_documents([text])
        chunks = [{'index': i+1, 'content': doc.page_content} for i, doc in enumerate(docs)]

        # 2. 向量化
        embeddings = get_embedding_model()
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})

        # 3. 检索
        retrieved_docs = retriever.invoke(query)
        retrieved = [{'index': i+1, 'content': doc.page_content} for i, doc in enumerate(retrieved_docs)]

        # 4. 生成答案
        os.environ["OPENAI_API_KEY"] = "sk-3a247053f2ab42bdb935539794b97231"
        os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question"],
                    template="""你是一个知识助手。请严格基于以下检索到的文档内容回答问题。

检索到的相关文档：
{context}

用户问题：{question}

回答要求：
1. 必须基于上述检索到的内容进行回答
2. 引用原文中的关键信息来支持你的回答
3. 如果检索内容无法回答问题，请明确说明"基于提供的文档无法回答此问题"
4. 不要添加检索内容之外的知识

请作答："""
                )
            }
        )
        result = qa.invoke(query)

        return jsonify({
            'success': True,
            'chunks': chunks,
            'chunks_count': len(chunks),
            'retrieved': retrieved,
            'retrieved_count': len(retrieved),
            'answer': result['result'],
            'source_docs': [doc.page_content for doc in result.get('source_documents', [])]
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_experiments():
    """对比多组实验"""
    data = request.json
    text = data.get('text', DEFAULT_TEXT)
    query = data.get('query', '合同订立应遵循哪些原则？')
    chunk_sizes = data.get('chunk_sizes', [20, 80, 200])

    results = []

    # 预加载 embedding 模型
    embeddings = get_embedding_model()

    # 预配置 LLM
    os.environ["OPENAI_API_KEY"] = "sk-3a247053f2ab42bdb935539794b97231"
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
    base_llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

    for chunk_size in chunk_sizes:
        try:
            # 1. 分片
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=2
            )
            docs = splitter.create_documents([text])

            # 2. 向量化
            db = FAISS.from_documents(docs, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 2})

            # 3. 检索
            retrieved_docs = retriever.invoke(query)

            # 4. 生成答案
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""你是一个知识助手。请严格基于以下检索到的文档内容回答问题。

检索到的相关文档：
{context}

用户问题：{question}

回答要求：
1. 必须基于上述检索到的内容进行回答
2. 引用原文中的关键信息来支持你的回答
3. 如果检索内容无法回答问题，请明确说明"基于提供的文档无法回答此问题"
4. 不要添加检索内容之外的知识

请作答："""
            )

            qa = RetrievalQA.from_chain_type(
                llm=base_llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            result = qa.invoke(query)

            results.append({
                'chunk_size': chunk_size,
                'chunks_count': len(docs),
                'chunks': [doc.page_content for doc in docs],
                'retrieved': [doc.page_content for doc in retrieved_docs],
                'retrieved_count': len(retrieved_docs),
                'answer': result['result'],
                'success': True
            })

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            results.append({
                'chunk_size': chunk_size,
                'success': False,
                'error': str(e),
                'detail': error_detail
            })

    return jsonify({'results': results})

if __name__ == '__main__':
    print("=" * 60)
    print("RAG 分片对比实验演示系统启动中...")
    print("=" * 60)
    print("\n请在浏览器中访问：http://localhost:5002")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5002)
