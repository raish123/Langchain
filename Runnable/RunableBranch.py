#Runablebranch class generally we used to implement condition in pipeline
from langchain_core.runnables import RunnableSequence, RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableBranch
from langchain_openai import  ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field,computed_field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal

load_dotenv()

parser = StrOutputParser()

# Hugging Face endpoint
llm_check_point = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)

# Chat model wrapper
model1 = ChatHuggingFace(llm=llm_check_point)
model2 = ChatOpenAI(temperature=0.5)
model3 = ChatGoogleGenerativeAI(model='gemini-2.5-pro')


#creating schema that i want from llm forcefully generate structure manner 
#that will be creating by using pydanticoutputparser class
class EmailClassifier(BaseModel):
    sentiment:Literal["complain","general query",'refund'] = Field(...,description="generate the sentiment either one complain,general query,refund ")


#creating an object of pydantic parser
pydantic_parser = PydanticOutputParser(pydantic_object=EmailClassifier)


email_query = """
The product is not functioning as expected ,
I would like more details about its warranty.I would like to request a refund due to dissatisfaction.
"""

#designing the dynamic prompt templates that we are passing to the model
#taking the email input from user fir us input ko prompt template ready karengey so that model will generate suitable output.
prompt_user = PromptTemplate(
    template="Generate the sentiment from user {input}.\n{format_instruction}",
    input_variables=['input'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)


#designing sequential chain by using sequeneRunnable to get sentiment
sentiment_chain = RunnableSequence(prompt_user,model2,pydantic_parser)


#if sentiment is coming out either one ["complain","general query",'refund'] from list
#then we have to design prompt accordingly before passing to models

# Response prompts for each sentiment
prompt_complain = PromptTemplate(
    template="Generate a one-line {complaint} response: Apologize politely and assure support.",
    input_variables=['complaint']
)

prompt_query = PromptTemplate(
    template="Generate a one-line {query_response}: Provide polite clarification.",
    input_variables=['query_response']
)

prompt_refund = PromptTemplate(
    template="Generate a one-line {refund} response: Politely inform about refund initiation.",
    input_variables=['refund']
)



#now creating branch Runnable to execute individual chain based on condition.
branch_chain = RunnableBranch(
    (lambda x : x.sentiment == "complain",RunnableSequence(prompt_complain,model1,parser)),
    (lambda x : x.sentiment == "refund",prompt_refund|model2|parser),
    (lambda x : x.sentiment == "query_response",prompt_query|model3|parser),
    RunnableLambda(lambda x : "Could not classify the email type.")
)



#now combining both chain sequential order
final_chain = RunnableSequence(sentiment_chain,branch_chain)

#invoking final chain
response = final_chain.invoke(input={'input':email_query})

print(response)





"""
we can do this manner also

# user defined functions for conditions
def is_complain(x):
    return x.sentiment == "complain"

def is_refund(x):
    return x.sentiment == "refund"

def is_query(x):
    return x.sentiment == "general query"   # 👈 aapke case me 'general query' hona chahiye
    
branch_chain = RunnableBranch(
    (is_complain, prompt_complain | model1 | parser),
    (is_refund, prompt_refund | model2 | parser),
    (is_query, prompt_query | model3 | parser),
    RunnableLambda(lambda _: "Could not classify the email type.")
)




"""