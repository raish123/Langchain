#This loader we used to fetch text content from static website(backend mei they using  BEAUTIFULSOUP and html parser to parse tags)

from langchain_community.document_loaders import WebBaseLoader
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


#creating an object of webbase loader
loader = WebBaseLoader("https://www.imdb.com/chart/top/") #we can also pass the lst of URLS
docs = loader.load()

print(docs[0].page_content)
print('*'*50)


prompt = PromptTemplate(
    template="You are very helpful Assistant,Please show me top 5 movie from given {text} text.",
    input_variables=['text'],
    
)


chain = RunnableSequence(prompt,model,parser)

response = chain.invoke(input={'text':docs[0].page_content})
print(response)