import os
import time
import signal
from openai import OpenAI
from typing import List
from langchain_community.vectorstores import Zilliz
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.document_loaders import Docx2txtLoader  
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# ModelScope相关导入
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from langchain_community.embeddings import ModelScopeEmbeddings

# ---- Windows兼容性补丁 ----
if not hasattr(signal, 'SIGALRM'):
    signal.alarm = lambda x: None  # 防止因缺少SIGALRM崩溃

# Configuration - 更新为ModelScope的配置
CONFIG = {
    "zilliz": {
        "endpoint": "https://in03-ffc7c224e8d79fe.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
        "user": "db_ffc7c224e8d79fe",
        "password": "Tu2&R*ns84q[.jah"
    },
    "modelscope": {
        "api_base": "https://api-inference.modelscope.cn/v1/",
        "api_key": "8a9000ef-331b-41c2-9a03-32020b8e5986",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "embedding_model": "iic/nlp_gte_sentence-embedding_chinese-base"
    }
}

class ModelScopeLLM:
    def __init__(self, api_base, api_key, model_id):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model_id = model_id
        
    def __call__(self, prompt, **kwargs):
        stream = kwargs.get("stream", False)
        
        try:
            if stream:
                return self._stream_response(prompt, **kwargs)
            else:
                return self._get_response(prompt, **kwargs)
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return "抱歉，我无法回答这个问题。"
    
    def _get_response(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False,
            max_tokens=kwargs.get("max_length", 512),
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content
    
    def _stream_response(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            max_tokens=kwargs.get("max_length", 1024),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        full_response = ""
        done_reasoning = False
        
        for chunk in response:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content or ""
            answer_chunk = chunk.choices[0].delta.content or ""
            
            if reasoning_chunk:
                print(reasoning_chunk, end='', flush=True)
            elif answer_chunk:
                if not done_reasoning:
                    print('\n\n === 最终回答 ===\n')
                    done_reasoning = True
                print(answer_chunk, end='', flush=True)
                full_response += answer_chunk
        
        return full_response

def get_file_loader(file_path: str):
    """根据文件扩展名返回适当的加载器"""
    try:
        if file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            return TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.docx'):
            return UnstructuredWordDocumentLoader(file_path, mode="elements")
        elif file_path.endswith('.pptx'):
            return UnstructuredPowerPointLoader(file_path)
        elif file_path.endswith('.xlsx'):
            return UnstructuredExcelLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)
    except Exception as e:
        print(f"创建文件加载器时出错: {str(e)}")
        return None

def load_documents_from_input():
    """根据用户输入加载文档"""
    print("\n请选择文档来源：")
    print("1. 网页URL")
    print("2. 本地文件")
    print("3. 两者都有")
    choice = input("请输入选择(1/2/3): ").strip()

    docs = []
    text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)

    if choice in ['1', '3']:
        urls = input("请输入网页URL(多个URL用逗号分隔): ").strip().split(',')
        urls = [url.strip() for url in urls if url.strip()]
        if urls:
            print("正在加载网页内容...")
            try:
                web_loader = WebBaseLoader(urls)
                web_docs = web_loader.load()
                docs.extend(text_splitter.split_documents(web_docs))
                print(f"成功加载 {len(urls)} 个网页")
            except Exception as e:
                print(f"加载网页时出错: {str(e)}")

    if choice in ['2', '3']:
        file_paths = input("请输入文件路径(多个文件用逗号分隔): ").strip().split(',')
        file_paths = [path.strip() for path in file_paths if path.strip()]
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，已跳过")
                continue
                
            print(f"正在加载文件: {file_path}...")
            start_time = time.time()
            
            try:
                # 根据文件类型选择加载器
                if file_path.lower().endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                elif file_path.lower().endswith('.pdf'):
                    from langchain.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                elif file_path.lower().endswith('.txt'):
                    from langchain.document_loaders import TextLoader
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    print(f"不支持的文件类型: {file_path}")
                    continue
                    
                file_docs = loader.load()
                split_docs = text_splitter.split_documents(file_docs)
                docs.extend(split_docs)
                elapsed = time.time() - start_time
                print(f"成功加载 {len(split_docs)} 个文档块 (耗时: {elapsed:.2f}秒)")
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    if not docs:
        print("警告: 没有加载任何文档，将使用默认文档")
        try:
            default_loader = WebBaseLoader(["https://milvus.io/docs/overview.md"])
            default_docs = default_loader.load()
            docs = text_splitter.split_documents(default_docs)
            
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.setdefault('source', 'default_source')
                doc.metadata.setdefault('title', 'default_title')
        except Exception as e:
            print(f"加载默认文档时出错: {str(e)}")
            raise RuntimeError("无法加载任何文档，请检查输入")

    return docs

def initialize_components():
    """Initialize embeddings, vector store, and LLM components using ModelScope."""
    print("\n初始化系统中...")
    start_time = time.time()
    
    try:
        # 设置 ModelScope token
        os.environ['MODELSCOPE_API_TOKEN'] = CONFIG["modelscope"]["api_key"]
        
        docs = load_documents_from_input()
        print(f"文档加载完成，共 {len(docs)} 个文档块")
        
        # Initialize ModelScope embeddings
        print("初始化ModelScope嵌入模型...")
        embeddings = ModelScopeEmbeddings(
            model_id=CONFIG["modelscope"]["embedding_model"]
        )
        
        # 连接到Zilliz
        connections.connect(
            alias="default",
            uri=CONFIG["zilliz"]["endpoint"],
            user=CONFIG["zilliz"]["user"],
            password=CONFIG["zilliz"]["password"],
            secure=True
        )
        
        # 定义集合名称
        collection_name = "LangChainCollection"
        
        # 检查并删除已存在的集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 定义schema - 使用更小的max_length确保安全
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),  # 更保守的长度限制
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="LangChain文档集合")
        
        # 创建集合
        collection = Collection(collection_name, schema)
        
        # 准备插入数据 - 更严格的分割策略
        processed_docs = []
        max_length = 30000  # 更小的分块大小，确保安全
        
        for doc in docs:
            if len(doc.page_content) > max_length:
                print(f"警告: 文档 '{doc.metadata.get('title', '无标题')}' 过长 ({len(doc.page_content)} 字符)，将被分割")
                
                # 更智能的分割方式，按句子或段落分割
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_length,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    new_metadata = doc.metadata.copy()
                    new_metadata["chunk_id"] = f"{i}"
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata=new_metadata
                    ))
            else:
                processed_docs.append(doc)
        
        # 再次检查所有文本长度
        for doc in processed_docs:
            if len(doc.page_content) > 32768:
                doc.page_content = doc.page_content[:32768]  # 硬截断作为最后手段
                print(f"警告: 文档 '{doc.metadata.get('title', '无标题')}' 仍然过长，已截断")
        
        texts = [doc.page_content for doc in processed_docs]
        metadatas = [doc.metadata for doc in processed_docs]
        
        # 分批处理嵌入
        batch_size = 32
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings_list.extend(embeddings.embed_documents(batch))
            except Exception as e:
                print(f"嵌入处理失败于批次 {i//batch_size}: {str(e)}")
                # 跳过问题批次或使用空向量
                embeddings_list.extend([[0.0]*768 for _ in batch])
        
        # 分批插入数据
        insert_batch_size = 50  # 更小的批次大小
        total_inserted = 0
        
        for i in range(0, len(texts), insert_batch_size):
            batch_texts = texts[i:i + insert_batch_size]
            batch_embeddings = embeddings_list[i:i + insert_batch_size]
            batch_metadatas = metadatas[i:i + insert_batch_size]
            
            # 确保所有文本长度合规
            batch_texts = [text[:32768] for text in batch_texts]
            
            entities = [
                batch_texts,
                batch_embeddings,
                [meta.get("source", "") for meta in batch_metadatas],
                [meta.get("title", "") for meta in batch_metadatas],
                batch_metadatas
            ]
            
            try:
                collection.insert(entities)
                total_inserted += len(batch_texts)
                print(f"已成功插入 {total_inserted}/{len(texts)} 个文档块")
            except Exception as e:
                print(f"插入失败于批次 {i//insert_batch_size}: {str(e)}")
                # 尝试逐条插入
                for j in range(len(batch_texts)):
                    try:
                        single_entity = [
                            [batch_texts[j]],
                            [batch_embeddings[j]],
                            [batch_metadatas[j].get("source", "")],
                            [batch_metadatas[j].get("title", "")],
                            [batch_metadatas[j]]
                        ]
                        collection.insert(single_entity)
                        total_inserted += 1
                    except Exception as single_e:
                        print(f"无法插入文档块 {i+j}: {str(single_e)}")
        
        collection.flush()
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("vector", index_params)
        collection.load()
        
        # 初始化向量存储
        vector_store = Zilliz(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={
                "uri": CONFIG["zilliz"]["endpoint"],
                "user": CONFIG["zilliz"]["user"],
                "password": CONFIG["zilliz"]["password"],
                "secure": True
            }
        )
        
        # Initialize ModelScope LLM
        # Initialize ModelScope LLM
        print("初始化ModelScope语言模型API...")
        llm = ModelScopeLLM(
            api_base=CONFIG["modelscope"]["api_base"],
            api_key=CONFIG["modelscope"]["api_key"],
            model_id=CONFIG["modelscope"]["model_id"]
        )
        
        elapsed = time.time() - start_time
        print(f"系统初始化完成 (总耗时: {elapsed:.2f}秒)\n")
        
        return vector_store, llm
        
    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        raise
            
def run_qa_loop(vector_store, llm):
    """Run interactive question answering loop."""
    prompt_template = """请根据以下上下文提供详细的中文回答。回答应该全面、准确且包含相关细节。
    
    上下文:
    {summaries}
    
    问题: {question}
    详细回答:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["summaries", "question"]
    )
    
    print("\n问答系统已就绪！输入您的问题或输入 'exit' 退出。")
    
    try:
        sample_docs = vector_store.similarity_search(" ", k=3)
        sources = set(doc.metadata.get('source', '未知来源') for doc in sample_docs)
        print("当前知识库包含以下内容来源:")
        print("- " + "\n- ".join(sources))
    except Exception as e:
        print("无法显示完整知识库信息:", str(e))
    
    print()
    
    while True:
        question = input("您的问题: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("感谢使用，再见！")
            break
        
        if not question:
            print("请输入有效问题。")
            continue
        
        try:
            docs = vector_store.similarity_search(question, k=7)
            
            input_text = PROMPT.format(
                summaries="\n".join([doc.page_content for doc in docs]),
                question=question
            )
            
            # 流式输出
            print("\n === 思考过程 === \n")
            response = llm.client.chat.completions.create(
                model=llm.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                stream=True,
                max_tokens=1024,
                temperature=0.7
            )
            
            done_reasoning = False
            full_response = ""
            for chunk in response:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content or ""
                answer_chunk = chunk.choices[0].delta.content or ""
                
                if reasoning_chunk:
                    print(reasoning_chunk, end='', flush=True)
                elif answer_chunk:
                    if not done_reasoning:
                        print('\n\n === 最终回答 ===\n')
                        done_reasoning = True
                    print(answer_chunk, end='', flush=True)
                    full_response += answer_chunk
            
            print("\n\n参考来源:", ", ".join(set(doc.metadata.get('source', '未知') for doc in docs)))
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")

def main():
    """Main execution function."""
    print("正在初始化系统，请稍候...")
    try:
        vector_store, llm = initialize_components()
        run_qa_loop(vector_store, llm)
    except Exception as e:
        print(f"系统运行失败: {str(e)}")
        print("建议：如果Qwen-7B无法运行，请尝试以下解决方案：")
        print("1. 确保已安装NVIDIA GPU驱动和CUDA")
        print("2. 尝试使用更小模型如Qwen-1.8B-Chat")
        print("3. 使用WSL或Linux环境运行")

if __name__ == "__main__":
    main()