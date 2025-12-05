import os

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
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
# Map-reduce
# 第一步：切割阶段，每个小的docs 为800个token，重复为0
text_splitter =  RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
split_docs= text_splitter.split_documents(docs)
# 第二步: Map 阶段
map_template = """以下是一组文档(documents)
"{docs}"
根据这个文档列表，请给出总结摘要
"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=model, prompt=map_prompt)

# 此时得到的是各个子文档的摘要

# 第三步： Reduce 阶段：分为中间环境的combine 和最终的reduce
reduce_template = """以下是一组总结摘要：
{docs}
将这些内容提炼成一个最终的、通义的总结摘要
"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)

# reduce 的思路：如果map之后的文档累计token也超过了限定的max_tokens，那我们将递归地将文档摘要以<=max_tokens的 批次传递给我们
# 大模型进行摘要总结，一旦这些批量摘要的累计大小小于 max_tokens，最后传递给大模型进行最后一次总结，以创建最终摘要

# 定义一个combine 的chain
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')

reduce_chain = ReduceDocumentsChain(
    # 这是最终调用的链
    combine_documents_chain=combine_chain,
    # 中间汇总的链
    collapse_documents_chain=combine_chain,
    # 将文档分组的最大令牌数
    token_max=3000  # 默认最大值是3000
)

# 第四步：合并所有的链
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name='docs',
    return_intermediate_steps=False  # 是否返回中间步骤的总结，默认值是False
)

# 第五步： 调用最终的链chain
result=map_reduce_chain.invoke(split_docs)
print(result)