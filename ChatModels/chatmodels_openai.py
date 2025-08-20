#first component in Langchain We are refering is Models
#using this models components we are interacting with 2 kinds of AI Models.
#a)LLM models b)chat models

#Note:-chat Models are specialized in Conversation


#importing the module.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

#creating an object of ChatOpenAI class 
chat_model = ChatOpenAI(
    model="gpt-4",temperature=0.9,max_completion_tokens=20
)


#now passing the input to llm models
response = chat_model.invoke(input='write single line poet on love')

print(response.content)
