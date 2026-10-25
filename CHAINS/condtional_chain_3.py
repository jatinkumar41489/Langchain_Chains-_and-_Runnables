from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser 
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    #max_output_tokens=1024,
    temperature=0.3,
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['Positive','Negative'] = Field(description = 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}', 
    
    input_variables = ['feedback'] ,
    
    partial_variables = { 'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

"""
Conditional Routing: RunnableBranch acts like an if-elif-else statement to dynamically choose which sub-chain to run based on the input data.

Fallback Safety: It evaluates conditions top-to-bottom and executes the first matching branch (positive or negative); if neither matches, it falls back to the default RunnableLambda function.
"""

branch_chain = RunnableBranch(
    (lambda x : x.sentiment.lower() == 'positive' , prompt2 | model | parser) , 
    (lambda x : x.sentiment.lower() == 'negative' , prompt3 | model | parser) ,
    RunnableLambda(lambda x : 'Could not find sentiment')
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a wonderful phone'}))

chain.get_graph().print_ascii()