from pydantic import BaseModel,Field,computed_field
from typing import Literal,Optional,Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json


load_dotenv()


#creating an object of ChatHuggingFace class
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)


user_query = """The smartphone's battery life is excellent and lasts two days easily. 
However, the camera quality is poor in low light. The user interface is smooth 
and responsive. Overall, I am happy with the purchase but wish the camera was better. and i give 3 rating"""


#creating pydantic class in that class whatever structured output(schema) we want it we r defining that.
class Review(BaseModel):
    summarization : str = Field(...,description="generate a  brief summary of the review")
    sentiments : Annotated[
        Literal["positive", "negative", "neutral"],Field(...,description="return sentiment of the review ")
        ]
    
    pros: str = Field(...,description="if dont find any pros in review return None")
    
    cons:str = Field(...,description="if dont find any cons in review return None")
    
    overall_rating: Annotated[
        Optional[int],
        Field(
            description="Rating on scale of 1-5 where 5 = best",
            ge=1,
            le=5,
        )]
    
    
#now wrapping the ReviewReviewParser class to llm model
llm_structure = chat_model.with_structured_output(Review)


#creating an object of Chatprompt template
prompt = ChatPromptTemplate(
    [
        ('system','you are good assistant who can clearly distinguish the review and generate summarization and sentiment pros and cons and rating dont find return none dont hallucinate'),
        ('human',"{user_query}")
    ]
)


#now filling the place holder value.
result = prompt.invoke({
    'user_query':user_query
})


#Get structured response
output: Review = llm_structure.invoke(result)



#showing the output in json
print("Output:\n", output.model_dump_json())