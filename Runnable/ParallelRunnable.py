#RunnableParallel Run multiple Task Parallely--->will take same inputs and return output as dictatonary format
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
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
model2  = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.5)

# Prompt 1
prompt1 = PromptTemplate(
    template="generate the tweet for this {topic}",
    input_variables=['topic']
)

# Prompt 2
prompt2 = PromptTemplate(
    template= "generate the linkedin post for this {topic}",
     input_variables=['topic']
)


parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1,model1,parser),
    'linkedin' : RunnableSequence(prompt2,model2,parser)
})

result = parallel_chain.invoke({
    'topic':'Artifical Intelligent'
})

print(result)