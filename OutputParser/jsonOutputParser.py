from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()


#creating dynamic prompt for single turn conversation using prompt template class!!!
prompt_temp = PromptTemplate(
    template="give me the name,age,city of a fictional person \n{format_instruction}",
    partial_variables={"format_instruction":parser.get_format_instructions()} #return the format instructions for the JSON output
)

prompt = prompt_temp.invoke({})
print(prompt)


#output prompt ka aisa dikhegha:- text='give me the name,age,city of a fictional person \nReturn a JSON object.'


#passing the prompt to llm model
response = chat_model.invoke(prompt)


#parsing the meta data using jsonoutputparser object
result = parser.parse(response.content)
print(result)


