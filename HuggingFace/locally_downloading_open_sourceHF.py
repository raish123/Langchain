#importing module that we can download the open source Fine tuned Base Model directly through HF and do inferencing accordingly.
#examples:-mistralai/Mistral-7B → Base pretrained model (next-token prediction, raw LM).
#mistralai/Mistral-7B-Instruct-v0.3 → Fine-tuned model (instructions follow karta hai, human-friendly outputs deta hai).

from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os


#want to save this pretrained model into specific folder.
os.environ['HF_HOME'] = "D:/huggingface_cache"

load_dotenv()

#creating an object HuggingFacePipeline class.
llm_pipeline = HuggingFacePipeline.from_model_id(  
model_id="mistralai/Mistral-7B-Instruct-v0.3",  #it will download mistral related all file and models
task="text-generation",  
pipeline_kwargs={"max_new_tokens": 100,'temperature':0.5},  
)


model = ChatHuggingFace(llm_pipeline)


#now calling builtin method invoke of ChatHuggingFace class in which we r passing input as prompt in it.
response = model.invoke(input="How to make money easily?")


print(response.content)
