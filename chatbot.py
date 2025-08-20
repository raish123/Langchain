#here we are creating a simple chatbot in which we are performing converstion between user and AI 
#as we know that Every call of LLM through API is stateless (dont remember the previous chat history)
#another issue is that if we save previous chat history we have to mention also (user/AI)kisne kya bheja hai 
#so that llm ko understanding mile futhure response dene mei ki user ne kya request bheja hai aur AI ne kya response diya 

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
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


#now storing the chat history 
chat_history = [
    SystemMessage(content="you are very helpful AI assistant")
]

#definining infinite loop for conversation 
while True:
    
    #Taking input prompt from user
    user_input = input("User: ")
    
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == "exit":
        break
    
    else:
        response = chat_model.invoke(chat_history)
        print('AI: ',response.content)
        chat_history.append(AIMessage(content=response.content))

#Bus ek issue hai is chatbot mei previous chat history AImodel pata nahi chal paa raha hai  (user/AI)kisne kya bheja hai 
print(chat_history)
        
