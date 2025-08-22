#Runnable Lamda -->using this wrapper we can change any python function to runnable function
#simply return:- the whatever input we are passing same as output we are getting without modifying it.
from langchain_core.runnables import RunnableSequence, RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_openai import  ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

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


#BELOW IS THE EXAMPLES
#simple question to apply RunnableLambda
# def text_split(text):
#     return len(text.split())


# #now converting this python fxn to runnable function.
# python_runnable = RunnableLambda(text_split)
# #since this a runnable we can invoke this runnbale object
# response = python_runnable.invoke(input='who is the prime minister of india')
# print(response)



prompt1 = PromptTemplate(
    template="write a joke for these {topic}",
    input_variables=['topic']
)



#developing sequential chains
joke_chain = prompt1 | model1 | parser    #this prompt return string of joke came through LLM models


#creating user define 
def joke_split(response):
    return len(response.split())


#this joke parallely execute (parallel chain take input sane and give output in dictatonary)
parallel_chain = RunnableParallel(
    {
        'joke_pass_through': RunnablePassthrough(), #dict return object depend upon input variable dtype
        'joke_count': RunnableLambda(joke_split)
        
    }
)


final_chain = RunnableSequence(joke_chain,parallel_chain)

# #since this a runnable we can invoke this runnbale object
response = final_chain.invoke(input={'topic':'sex'})
print(response)

