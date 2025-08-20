#samjho hume ek chatbot model banaya hai example:-flipkart ke liye
#user ne input diya ki mere order cancel hum mujhe refund kub milegha
#to AI ne next 2 to 3 business working day.
#lekin samjho ek din phir woh user aaya usne mera refund nahi aaya to AI uska javab kaise degha

#sab se previous chat history humko save karna hoga CLOUD STORAGE PE
#jab dubara user ka request aaya sab se phele previous chat history load karni hogi fir AI response degha

#Note:-
#Placeholders basically dynamic variables hote hain jo tum template(chatprompt template) me rakhte ho,
#aur later code me unko fill karte ho actual values ke saath. 


from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
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



#creating a chat template variable
chat_templates = ChatPromptTemplate(
    [
        ('system','you are a helpful customers support agent'),
        #previous chat history we put in MessagesPlaceholder mei!
        MessagesPlaceholder(variable_name="chat_history"),
        ('human',"{query}")
    ]
)

chat_history = []
#calling the previous chat history
with open('Messages_Invoke_Usage\chat_history.txt') as file:
    content = file.readlines() #rtn entire line rtn as list object
    chat_history.extend(content)
    
    
    
#now filling the placeholder by using invole method
result = chat_templates.invoke(
    {
        "chat_history":chat_history,
        'query': "where is my refund waited two day"
    }
)


print(chat_history)
response = chat_model.invoke(result)
print(response.content)


