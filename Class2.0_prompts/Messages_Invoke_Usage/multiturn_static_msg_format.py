#Langchain Solve the Problem they store previous Chat History in keyward Argument Manner Mei!!!!

#SystemMessage LangChain (aur OpenAI API) me ek special type ka message 
#hota hai jo LLM ko instructions deta hai ki wo kaise behave kare.
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
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


#creating list of message.
messages = [
    SystemMessage(content="You are helpful assistant"),#this is the first meaage we r passing to llm to instruct llm Model how to behave
    HumanMessage(content="who is the prime minister of india?")

]


response = chat_model.invoke(messages)
messages.append(AIMessage(content=response.content))

print(messages) #printing the chat history of messages