from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


load_dotenv()

#first we have to define kon se HuggingFaceEndpoint pe API Request jayeghi
# Hugging Face endpoint define karo
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=128,
    do_sample=False,
)

# Chat model object banao
chat_model = ChatHuggingFace(llm=llm, verbose=True)


#now creating pydantic schema class in which we r defining class attibute.
class Person(BaseModel):
    name: str = Field(default=None,description='Name of the person')
    age: int = Field(default=None,description="age of the person",gt=18)
    city:str = Field(default=None,description="name of the person belongs to")
    
    
#now passing these schema class to pyandanticoutputparser class.
parser = PydanticOutputParser(pydantic_object=Person)


#creating dynamic prompt for single turn conversation using prompt template class!!!
prompt_temp = PromptTemplate(
    template="Generate the name,age and city of a fictional {place} person \n{format_instruction}",
    input_variables=['place'],
    partial_variables={"format_instruction":parser.get_format_instructions()} #return the format instructions for the pydantic object format.
)

    
prompt = prompt_temp.invoke({'place':'russia'}) #rtn the dynamic prompt.


result = chat_model.invoke(prompt) #rtn the response along with meta data

#parsing the reponse along with meta data
print(prompt)
final_output = parser.parse(result.content)
print(final_output)

