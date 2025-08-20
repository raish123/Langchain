#importing module that we connect to open source Fine tuned Base Model directly through HF Api and do inferencing accordingly.
#examples:-mistralai/Mistral-7B → Base pretrained model (next-token prediction, raw LM).
#mistralai/Mistral-7B-Instruct-v0.3 → Fine-tuned model (instructions follow karta hai, human-friendly outputs deta hai).

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

#first we have to define kon se HuggingFaceEndpoint pe API Request jayeghi
llm = HuggingFaceEndpoint(  
repo_id="mistralai/Mistral-7B-Instruct-v0.3",  
task="text-generation",
max_new_tokens=512,  
do_sample=False,  

)  


#creating an object of ChatHuggingFace class
chat_model = ChatHuggingFace(llm=llm,verbose=True)


#now calling builtin method invoke of ChatHuggingFace class in which we r passing input as prompt in it.
response = chat_model.invoke(input="How to make money easily?")


print(response.content)