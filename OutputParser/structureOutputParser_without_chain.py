from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema


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



#we are creating structure schema with help of ResponseSchema class
#define list of structure schema.
schema = [
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic'),
 ]


#creating a parser object
#for creating StructuredOutput we have to used built in method from_structur_response of StructuredOutputParser class.
parser = StructuredOutputParser.from_response_schemas(schema)



#creating dynamic prompt for single turn conversation using prompt template class!!!
prompt_temp = PromptTemplate(
    template="give me 3 facts about  {topic}. \n{format_instruction} ",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()} #return the format instructions for the structure output
)

prompt = prompt_temp.invoke({'topic':'sex'})


result = chat_model.invoke(prompt)



#now parsing this llm respons
response = parser.parse(result.content)

print(response)