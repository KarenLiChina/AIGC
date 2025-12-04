import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

# 创建模型
model = ChatOpenAI(model=os.getenv("MODEL_NAME"),
                   api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("BASE_URL"),
                   temperature=0.8)


# 生成结构化的数据：5个步骤
# 1. 定义数据模型
# 此数据模型中，注释并不能提供给ai来进行分析，没有给出Field的描述，后续可以通过少量样例供ai分析，所有字段非optional类型，需要AI都生成
class MedicalBilling(BaseModel):
    patient_id: int  # 患者id，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_core: str  # 诊断代码，字符串类型
    procedure_core: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2. 提供一些样例数据给AI，可以是3-10条少量案例，每个数据都需要有值
examples = [
    {
        "example": "Patient ID: 123456, Patient Name: 张三, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: 500 ,  INSURANCE CLAIM AMOUNT: 300"
    },
    {
        "example": "Patient ID: 789012, Patient Name: 李四, Diagnosis Code: M54.4, Procedure Code: 99213, Total Charge: 400 ,  INSURANCE CLAIM AMOUNT: 240"
    },
    {
        "example": "Patient ID: 345678, Patient Name: 王五, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: 760 ,  INSURANCE CLAIM AMOUNT: 539"
    }
]

# 3. 创建一个提示模板，用来指导AI 生成符合规定的数据
ex_prompt = PromptTemplate(input_variables=['example'],
                           template="{example}")  # input_variables 中的'example'是指从模板中读取什么样的参数，template 中的"{example}"表示模板中定义参数

# 定义少量样本的模板
prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,  # 前缀，少量样本的前缀
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,  # 后缀，少量样本的后缀
    examples=examples,  # 样本
    example_prompt=ex_prompt,  # example的模板
    input_variables=['subject', 'extra']  # 在最终生成的完整提示词模板中需要用户动态提供的变量，subject 生成数据的主题，extra 额外的要求 是两个占位符
)

# 4. 创建一个结构化的数据生成器, 很像 chain，但是输出结果为数据
generator = create_openai_data_generator(
    output_schema=MedicalBilling,  # 指定输出数据的格式
    llm=model,
    prompt=prompt_template
)
# 5. 调用生成器，不调用invoke，调用generate 方法生成数据
result = generator.generate(
    subject='医疗账单',  # 和input_variables 中的subject 保持一致，指定生成数据主题
    extra='患者姓名可以是随机的，尽量使用比较生僻的人名。账单总费用符合正态分布的正数，保险索赔金额最大不能超过总账单费用的70%。', # 和input_variables 中的extra 保持一致，额外的指导信息
    runs=10  # 指定的生成数据数量
)

print(result)
