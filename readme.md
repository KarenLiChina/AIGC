# python环境要求
```bash
pip install -r requirements.txt
```

# 配置环境变量

创建 `.env`文件，在.env文件中的设置 `MODEL_NAME`,`OPENAI_API_KEY` 和 `BASE_URL` 为自己的 key 和 url
`LANGCHAIN_TRACING_V2`设置为true，`LANGCHAIN_PROJECT`设置为项目名称，不配置默认为default，`LANGCHAIN_API_KEY`设置为LangSmith的API Key，可以在LangSmith中查看调用大模型使用情况，不需要也可以不配置这两个变量

## langsmith的检测数据
配置`LANGCHAIN_TRACING_V2`，`LANGCHAIN_API_KEY`后可以在https://smith.langchain.com/ Tracing Projects中查看调用大模型的使用情况

# 具体实现

## ai生成文本数据，下载依赖包langchain_experimental
ai_generate_text_data.py

## 生成结构化数 
ai_generate_structure_data.py


## 提取结构化的文本分类属性
text_classification.py

## 从文档中生成文本摘要，需要安装依赖包 chromadb,tiktoken
### 总结或组合文档的三种方式：
1. 填充(stuff)，简单地将文档一次性的放到prompt中发给大模型处理，当文本过大时，会失败
2. 映射-规约（Map-reduce），将文档分成多个小块，每块可以并行或串行发给大模型处理，生成摘要（map过程），然后将子结果再根据大模型的最大token数量进行分割，进行分析总结，将生成的子结果再反复进行组合分析（combine过程），直到最后一次的子结果发给大模型进行最终总结（reduce），合成最终统一的答案，理论上文本长度无上限 
3. 细化（Refine），通过顺序迭代文档来更新滚动摘要，先生成第一个文档块的摘要，然后将这个摘要和下一个文档块一起交给大模型处理，不断迭代指导最后得出答案。连贯性优于Map-reduce。
summary_document_stuff.py
summary_document_map_reduce.py
summary_document_refine.py