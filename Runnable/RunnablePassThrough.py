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

# Prompt 1: Answer the question
prompt1 = PromptTemplate(
    template="Write a joke about this {question}",
    input_variables=['question']
)

# Prompt 2: Explain the joke
prompt2 = PromptTemplate(
    template="Explain the following joke {topic} in single line",
    input_variables=['topic']
)


#developing sequential chain
joke_gen_chain = RunnableSequence(prompt1,model1,parser) #rtn as string text output

#if i have used input variable prompt template required input to be dictonary formed
map_input_chain = RunnableLambda(lambda x : {'topic':x})

joke_explain_chain = RunnableSequence(prompt2,model2,parser)

##developing parallel chain
parallel_chain = RunnableParallel(
    {
        'explanation': RunnableSequence(joke_gen_chain,map_input_chain,joke_explain_chain),
        'joke':RunnablePassthrough()
    }
)

response = parallel_chain.invoke({'question':'elephant'})
print(response)