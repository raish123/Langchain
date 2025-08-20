#importing modules
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()


#creating an object of OpenAIEmbeddings class 
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)




#assuming this is the document Page or Pdf pages 
#we have now we r converting them into embedding vector storing them into vector database!!!

cricket_players = [
    # Virat Kohli
    "Virat Kohli is known as the 'Run Machine' of modern cricket.He is famous for his consistency and aggressive style of play.",
    
    # Rohit Sharma
    "Rohit Sharma is called the 'Hitman' for his explosive batting.He is the only player to score three double centuries in One Day Internationals.As a captain, Rohit has led India to several victories.",

    # Jasprit Bumrah
    "Jasprit Bumrah is one of the most dangerous fast bowlers in the world.His yorkers in the death overs are almost unbeatable.Bumrah's unique bowling action has made him stand out in cricket."
]


user_query = "tell me about virat kohli"



#converting document into embedding vector storing them into vector base mei.
doc_emb_vector = emb_model.embed_documents(cricket_players)


#converting user_query input to embedding vector.
query_emb_vec = emb_model.embed_query(user_query)



#now searching for semantic search by using Cosine similarity.
similarity_score = cosine_similarity([query_emb_vec],doc_emb_vector)[0] #dono value ki same dimension honi chahiye query_emb_vec 1d tha changing them --->2D
#similarity_score return as 2d list object making them 1dimension.



indexes,similarity = sorted(list(enumerate(similarity_score)),key=lambda x: x[1])[-1] #rtn as a tuple object which havoing 2 things index,similarity


print(user_query)
print(cricket_players[indexes])