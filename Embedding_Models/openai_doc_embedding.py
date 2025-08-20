#embedding models we used to convert input_query or prompt
#to embedding vectors (this vector is known as conceptual vector or task specific vector)


#importing module which is used to perform embedding vectors(futhure we used to perform semantic search)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

#instead of this document object we can pass raw text pages content to get embedding vectors
document = [
    "delhi is the capital of india",
    "west bengal is capital of kolkata",
    "mumbai is a capital of maharashtra"
]


#creating an object of OpenAIEmbeddings class 
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=20
)


#now calling the built in method embed_documents we pass document prompt to convert into embedding vector.
emb_vector = emb_model.embed_documents(document)

print(emb_vector)