#import module 
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


model_name = "Qwen/Qwen3-Embedding-0.6B"  
model_kwargs = {'device': 'cuda'}  
encode_kwargs = {'normalize_embeddings': False}  

#creating an object of HuggingFaceEmbeddings class

hf_emb = HuggingFaceEmbeddings(  
model_name=model_name,  
model_kwargs=model_kwargs,  
encode_kwargs=encode_kwargs  
)


#calling the builtin method embed_query to convert single prompt to embeddding vector
hf_vect = hf_emb.embed_query(text = 'how to make money easily?')

print(hf_vect)


#Note:- Same chiz we can do it for document ke liye bhi one change is that 
#instead of using emb_query you have to used embed_documents method