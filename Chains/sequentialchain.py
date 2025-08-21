#In these sequential chain mei 
#we pass the input_prompt two time to model like given below.
#prompt--->llm--->output along with meta data(we as to get detailed report)--->string parser(input)--->llm--->response-->parser karke result show karengey

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



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


#Designing the prompt(single textual string input) that we r passing to model by using PromptTemplate class.
template1 = PromptTemplate(
    template="You are a helpful assistant.please give me detailed report {report}",
    input_variables=['report']
)


template2 = PromptTemplate(
    template="please generate 5 line summarize detail {text}",
    input_variables=['text']
)


#creating an object stroutputparser class
parser = StrOutputParser()

#forming to perform sequnetial operation
chain = template1 | chat_model | parser | template2 | chat_model | parser

#if i want to visualize the chains
#chain.get_graph().print_ascii()



#invoking or calling the model
response = chain.invoke({
    'report':'malaria'
})

print(response)

