from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableBranch,RunnableParallel,RunnablePassthrough,RunnableSequence
#with help of this runnable we gonna create chain of pipelines
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser


load_dotenv()


#creating an object of HuggingFaceEndpoint in this class we r defining the endpoint
#of model where we are sending request and getting response from it

llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    task='text-generation'

)

#creating an llm model object
model = ChatHuggingFace(llm = llm_endpoint)


#creating an object of stroutput parser
parser = StrOutputParser()



#creating an object of TextLoader
loader = TextLoader(
    file_path="DocumentLoaders\sample.txt",
    encoding="utf-8"
)


#now loading the content of text file.

data = loader.load() #rtn as list object
print(data[0].metadata)
print(data[0].page_content)
print('*'*50)


#creating single turn prompt templates object.
prompt = PromptTemplate(
    template="You are very helpful Assistant,Please Summarize these {topic} topic.",
    input_variables=['topic'],
    
)

#creating sequential by using runnable sequence
seq_chain = RunnableSequence(prompt,model,parser)



#invoking the response from LLM by passing this prompt
response = seq_chain.invoke(input={'topic':data[0].page_content})
print(response)