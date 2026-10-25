"""
Step 1 : User's Input
Step 2 : User's Input ---> LLMs
Step 3 : LLMs ----> Generate Response

FLOW : Prompt --> Model --> Parser
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 real application about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    #max_output_tokens=1024,
    temperature=0.3,
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke(
    {
        "topic":"AI"
    }
)

print(result)

chain.get_graph().print_ascii() # By this we can analyze the chain's workflow

