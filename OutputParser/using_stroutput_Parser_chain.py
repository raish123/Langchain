#importing the models using Langchain Framework
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#single turn message ko dynamically we can create PromptTemplate class 
#multi turn message ko dynamically we can create ChatPromptTemplate class 
from dotenv import load_dotenv

#now loading the environment variable to this file.
load_dotenv()

#creating an object of ChatGoogleGenerativeAI class.
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',temperature=0.5
)




#single turn meassage dynamically we are creating using Prompt template class in which we can pass place holder.
prompt1 = PromptTemplate(
    #template parameter we are defining message that we are passing to LLM models
    template="write a detailed report on this {topic}",
    input_variables=['topic']
)



#second dynamic propmpt creating
prompt2 = PromptTemplate(
    #template parameter we are defining message that we are passing to LLM models
    template="write a 5 line summary on this following text. /n{text}",
    input_variables=['text']
)


#now creating an object of Stroutput parser class.
parser  = StrOutputParser() 
#parser kya karegha jo response llm se aaya sirf usmei se string or text nikal ke pass karegha next chain block.


#creating a chain of pipeline will execute the workflow.
chain = prompt1 | model | parser | prompt2 | model | parser


#simply chain pipeline ko invoke karna hai
response = chain.invoke(
    {
        "topic" : "blackhole"
    }
)


print(response)