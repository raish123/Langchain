from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda


load_dotenv()

# Hugging Face endpoint define karo
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=128,
    do_sample=False,
)

#creating object of strouput parser class
parser = StrOutputParser()

# Chat model objects
model1 = ChatHuggingFace(llm=llm)
model2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
model3 = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)


# Problem: We don’t have full control over the LLM’s output. 
# For example, if we ask for sentiment, the model might return  text along with sentiment 
# like "The sentiment of this text is mostly positive" instead of just "positive".
# To solve this, we need to structure the output using a Pydantic output parser.

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field,BaseModel
from typing import Literal


#creating a pydantic schema class which ensure llm output should be this attribute formed.
class Review(BaseModel):
    sentiment:Literal['positive','negative'] = Field(...,description='A single-line response to the feedback')
    

#creating pydanticoutput parser object
pydantic_parser = PydanticOutputParser(pydantic_object=Review)



# Prompt templates
prompt1 = PromptTemplate(
    template="You are a helpful assistant. Classify the sentiment of the following feedback text into positive or negative Only.\n{feedback} \n{format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':pydantic_parser.get_format_instructions()} #rtn output as pydantic object
)


classify_chain = prompt1 | model1 | pydantic_parser


feedback_text = """I recently purchased a laptop from your store. 
The performance of battery life is very Bad.
However, the delivery was delayed by three days and the packaging was not secure.
Overall, I am not satisfied with the product but the delivery experience needs improvement."""


#designing another prompts.
#by using Prompt templates
prompt2 = PromptTemplate(
    template="You are a customer support assistant. Write a single-line empathetic response to this positive feedback. \n{feedback}",
    input_variables=['feedback'],
    
)

prompt3 = PromptTemplate(
    template="You are a customer support assistant. Write a single-line empathetic response to this negative feedback. \n{feedback}",
    input_variables=['feedback'],
)


#now creating branch chain object by using RunnableBranch class
#below is the sample syntax:-
# branch_chain = RunnableBranch(
#     ('condition1','condition true pe konsa chain execute hoga'),
#     ('condition2','condition true pe konsa chain execute hoga')
# )



branch_chain = RunnableBranch(
    # agar sentiment positive hai
    (lambda x : x.sentiment == 'positive', prompt2 | model2 | parser),

    
    # agar sentiment negative hai
    (lambda x : x.sentiment == 'negative', prompt3 | model3 | parser),
    
    # agar koi sentiment match na kare toh default chain bana raha hu using RunnableLambda class
    RunnableLambda(lambda x : "⚠️ could not find sentiment")
)



#creating merge chain
merge_chain = classify_chain | branch_chain

response = merge_chain.invoke({"feedback":feedback_text})

print(response)

merge_chain.get_graph().print_ascii()