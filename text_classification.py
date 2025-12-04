import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"),
                   temperature=0)  # 分类时，不要让模型提供更多的多言行，严谨输出


class Classification(BaseModel):
    """
    数据模型类，定义一个Pydantic的数据类型，未来需要根据该类型，完成文本分类
    """
    # 文本的情感倾向，预期为字符串类型，一般用Literal 来限定 可选值，不在field中使用 枚举
    sentiment: Literal['正面', '负面', '中性'] = Field(description="文本的情感分类")

    # 文本的攻击性，预期为1到5的整数
    aggressiveness: int = Field(description="描述文本的攻击性，数字越大表示攻击性越强")

    # 文本使用语言，预期为字符串类型
    language: str = Field(description="文本使用语言")


# 创建一个用于提取信息的提示词模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落提取所需信息。只提取'Classification'类中提到的属性。
    段落：
    {input}
    """
)

chain = tagging_prompt | model.with_structured_output(Classification, method='function_calling')
text = '谢逊神情日渐反常，眼睛中射出异样光芒，常自指手划脚的对天咒骂，胸中怨毒，竟自不可抑制。血红的太阳停在西边海面，良久良久，始终不沉下海去。谢逊突然跃起，指着太阳大声骂道：“连你太阳也来欺侮我，贼太阳，鬼太阳，我若是有张硬弓，一枝长箭，嘿嘿，一箭射你个穿。”突然伸手在冰上一击，拍下拳头大的一块冰，用力向太阳掷了过去。三人先后来到冰火岛，言及能不能回归中原，张翠山道：“那得瞧老天爷的意旨了。”谢逊破口骂道：“甚么老天爷，狗天、贼天、强盗老天！”'
result = chain.invoke({'input': text})  # 调用的返回结果是结构化的
print(result)
