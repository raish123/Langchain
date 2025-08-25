from langchain_community.document_loaders import PDFPlumberLoader
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

#creating an object of PDFPlumberLoader class
pdf_loader = PDFPlumberLoader(
    file_path="DocumentLoaders\LangChain.pdf",
)


#using pdf_loader object loading the textual content from pdf .
page_content = pdf_loader.load() #rtn list object of content from pages
print(page_content[0].page_content)