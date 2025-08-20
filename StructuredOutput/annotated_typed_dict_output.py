#Annotated typing we used to define meta data in it
from typing import Annotated,Literal,Optional,TypedDict
#creating a class who inheriting the property from parent class TypedDict 
#with help of them we are defining class variable

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


#creating an object of ChatHuggingFace class
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)



#defining the class variable how the structure output i want it!!!
class ReviewParser(TypedDict):
    summarization: Annotated[str,"generate a  brief summary of the review"]
    sentiment: Annotated[str,"return sentiment of the review either positive negative or neutral"]
    
    
    
#now wrapping the ReviewParser class to llm model
llm_structure = chat_model.with_structured_output(ReviewParser)



#review text
review_text = """The hardware is great, but the software feels bloated. 
There are too many pre-installed apps that I can't remove. Also,
the UI looks outdated compared to other brands. Hoping for a software update to fix this."""

#creating an object of Chatprompt template
prompt = ChatPromptTemplate(
    [
        ('system','you are good assistant who can clearly distinguish the review and generate summarization and sentiment'),
        ('human',"{review_text}")
    ]
)


#Fill placeholder with user_query value
result = prompt.invoke({
    'review_text':review_text
})



#now invoking the reponse from llm model
output:ReviewParser = llm_structure.invoke(result)


#Get structured response
print(prompt)
print("Full dict output:\n", output)

