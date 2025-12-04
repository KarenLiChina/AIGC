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