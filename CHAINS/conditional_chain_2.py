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

result = classifier_chain.invoke({'feedback' : 'This is a wonderful smartphone'}).sentiment

"""
This portion is done and to complete this we need to create branching that you can see in condtional_chain_3.py 
                  ┌──────────────────┐
                  │     Feedback     │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │      Model       │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │     Positive    │       │     Negative    │
     │      Model      │       │      Model      │
     └────────┬────────┘       └────────┬────────┘
     
"""

