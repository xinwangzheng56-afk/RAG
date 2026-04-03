import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

# ===== 1. 原始文档 =====
text = """
合同应当遵循公平原则,确定各方的权利和义务。
当事人依法享有自愿订立合同的权利,任何单位和个人不得非法干预。
当事人应当遵循诚实信用原则,履行义务。
"""

query = "合同订立应遵循哪些原则?"

# ===== 2. 实验函数 =====
def run_experiment(chunk_size):
    print(f"\n===== chunk_size = {chunk_size} =====")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=10
    )

    docs = splitter.create_documents([text])

    print("\n--- 分片结果 ---")
    for i, doc in enumerate(docs):
        print(f"chunk{i+1}: {doc.page_content}")

    # 使用本地 embedding 模型
    print("\n--- 加载本地 embedding 模型 ---")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # LLM (使用 DeepSeek)
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
        retriever=retriever
    )

    # 检索内容
    print("\n--- 检索结果 ---")
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        print(doc.page_content)

    # 最终回答
    print("\n--- 模型回答 ---")
    result = qa.invoke(query)
    print(result['result'])


# ===== 3. 三组对比实验 =====
run_experiment(20)   # 小分片
run_experiment(80)   # 中分片
run_experiment(200)  # 大分片
