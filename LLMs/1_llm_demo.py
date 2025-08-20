#first component in Langchain We are refering is Models
#using this models components we are interacting with 2 kinds of AI Models.
#a)LLM models b)chat models
#in LLM models we have 2 kinds of models in it 1)base model LLM using this model 
#we can perform following task like summarisation,q/a task,sentiment analysis,text generation
#2)chat Models are specialized in Conversation

#importing the module that can load LLM's
from langchain_openai import OpenAI
from dotenv import load_dotenv


#calling the function of load_dotenv
load_dotenv()



#creating an object of OpenAI class that loading the LLM Base Models!!
#temperature is creative parameter based on the variance of temperature we can generate output.
llm_model = OpenAI(
    model="gpt-3.5-turbo-instruct",temperature=0.7
)


#calling the built in method invoke of OpenAI class to pass the query to llm models.
response = llm_model.invoke(input='what is the capital state of india')

print(response)