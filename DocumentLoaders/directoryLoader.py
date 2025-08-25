#In this file we are loading the data from root directory.
from langchain_community.document_loaders import DirectoryLoader,PDFPlumberLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableBranch,RunnableParallel,RunnablePassthrough,RunnableSequence
#with help of this runnable we gonna create chain of pipelines
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser


#creating directory loader object
directory = DirectoryLoader(
    path="DocumentLoaders\\New_folder",
    glob=["*.pdf"],
     show_progress=True,
     loader_cls=PDFPlumberLoader
    
)


# Lazy load generator
documents_gen = directory.lazy_load()

# Agar specific page/document chahiye toh list bana lo
documents_list = list(documents_gen)

# Ab access kar sakte ho
print(documents_list[30].page_content)
print(documents_list[30].metadata)


for i, doc in enumerate(directory.lazy_load()):
    print(f"\n--- Page {i+1} ---")
    #print(doc.page_content[:500])  # sirf 500 chars print karne ke liye
    print(doc.metadata)