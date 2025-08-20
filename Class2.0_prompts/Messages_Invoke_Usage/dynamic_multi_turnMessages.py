# LangChain ke e ChatPromptTemplate ke baare mein pooch rahe ho.
# Ye ek powerful cheez hai jo tumhe multi-turn chat prompts ko structured way me likhne aur reuse karne ka option deti hai.

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


#HuggingFaceEndpoint mei hum mentioned karte konsa model used karna hai or konse endpoint pe request jayeghi..
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=128,
    temperature=0.5
    
)


#creating an object chathugging face model.
chat_model = ChatHuggingFace(llm=llm,verbose=True)


prompt = ChatPromptTemplate([
    ('system','You are  expert in this {domain} domain need your experties'),
    ("human",'tell me about this  {dieseas}')
]
)


#now filling the place holder by using invole method of Chatprompt template class
response = prompt.invoke({
    'domain':'doctor',
    'dieseas':'malaria'
})

print(response)



#now passing this dynamic prompt to llm models
return_response = chat_model.invoke(response)
print(return_response.content)