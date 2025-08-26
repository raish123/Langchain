#importing the module
from langchain_openai import OpenAIEmbeddings #this class we used to load openai embedding models
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate #this class we used to create dynamic or role based prompt
import os
from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableSequence
from langchain_openai import ChatOpenAI #this class we used to load openai llm models
from langchain_chroma import Chroma #this class we used to create chroma vector store database
from langchain_community.document_loaders import TextLoader,PDFPlumberLoader,DirectoryLoader
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field,computed_field

load_dotenv()


#creating an object of embedding models
emb_models = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=128
)


#creating and object llm chatmodel 
chat_model = ChatOpenAI(
    temperature=0 #we called as creative parameter
)

# Custom Loader Class
class CustomTextLoader(TextLoader):
    def lazy_load(self):
        # File name extract
        filename = os.path.basename(self.file_path)   # e.g. "RCB_virat.txt"
        team_name = filename.split("_")[0]            # e.g. "RCB"
        
        # Iterate lazily from parent class
        for doc in super().lazy_load():
            doc.metadata["team"] = team_name
            yield doc


# DirectoryLoader with CustomTextLoader
loader = DirectoryLoader(
    path=r"VectorStore\textDocument",
    loader_cls=CustomTextLoader,
    show_progress=True,
    glob="*.txt"
)

# lazy load documents
documents = list(loader.lazy_load())
print(documents)


#now creating object chroma vector store database by using Chroma class.
chroma = Chroma(
    #embedding_function parameter we have to define embedding models who will convert doc to embedding vectors
    embedding_function=emb_models, 
    persist_directory= "chroma_db", #means which loaction u want to store embedding vectors
    collection_name="sample"
    
)



#Now adding the documents to vector database
chroma.add_documents(documents)



#if i want to show the documents from vector store database
chroma.get(include=["embeddings", "metadatas", "documents"])