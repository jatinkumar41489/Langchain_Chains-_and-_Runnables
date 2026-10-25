"""
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
              ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │ Positive Output │       │ Negative Output │
     └─────────────────┘       └─────────────────┘

Flow: Feedback → Model → Positive/Negative branches → Separate model processing → Corresponding output. 

"""
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    #max_output_tokens=1024,
    temperature=0.3,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into postive or negative \n {feedback} ', 
    input_variables = ['feedback']
)

classifier_chain = prompt1 | model | parser

print(classifier_chain.invoke(
    {
        'feedback' : 'This is a wonderful smartphone'
    }
))

"""
This time it is giving result as Positive or Negative but LLM can give result as The sentence is Positive or Negative. 
So, getting the output everytime as Positive or Negative we need to Structure the Output with the help of Pydantic Output Parser.
You can see in conditional_chain_2.py
"""