import os

from dotenv import load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"),
                   temperature=0)

loader = WebBaseLoader(web_paths=['https://blog.csdn.net/2301_82275412/article/details/148773003'])
docs = loader.load()  # 得到整篇文字
# Refine RefineDocumentsChain 类似于映射规约
# RefineDocumentsChain通过循环遍历输入文档逐步更新回复，对于每个分块分档，它将当前文档和最新的中间答案传递给LLM，以获得最新的总结。

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 第一种没有prompt的chain，指定chain_type 为refine，
chain = load_summarize_chain(model, chain_type='refine')

result = chain.invoke(split_docs)
print(result['output_text'])

# 第二种 有prompt的chain
prompt_template = """针对下面的内容，写一个简洁的总结摘要
"{text}"
"""
prompt = PromptTemplate.from_template(prompt_template)
# 其中existing_answer 为上一步的结果，是一个占位符，叫什么名字都可以
refine_template = """你的工作是做出一个最终的总结摘要，现有一个前文文档的现有摘要：{existing_answer},
    现在要结合前文摘要和后续的文本，生成最新的摘要内容：
    "{text}"
"""
refine_prompt = PromptTemplate.from_template(refine_template)
prompt_chain = load_summarize_chain(llm=model,
                                    chain_type='refine',
                                    question_prompt=prompt,
                                    refine_prompt=refine_prompt,
                                    return_intermediate_steps=False,
                                    input_key="input_documents",
                                    output_key="output_text"
                                    )
result= prompt_chain.invoke({"input_documents": split_docs},return_only_outputs=False)
print(result)