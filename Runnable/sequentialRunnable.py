from langchain_core.runnables import RunnableSequence, RunnableLambda
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
model = ChatHuggingFace(llm=llm_check_point)

# Prompt 1: Answer the question
prompt1 = PromptTemplate(
    template="You are a very helpful AI assistant. Write a joke about this {question}",
    input_variables=['question']
)

# Prompt 2: Explain the joke
prompt2 = PromptTemplate(
    template="Explain the following joke {topic} in single line",
    input_variables=['topic']
)

# RunnableSequence with commas
chain = RunnableSequence(
    prompt1,
    model,
    parser,
    RunnableLambda(lambda x: {"topic": x}),
    prompt2,
    model,
    parser
)

# Run
response = chain.invoke({"question": "nirmala sitaraman"})
print(response)





#Below is the another Way to perform Runnable Sequence Execution.


# # Chain banate waqt ensure karo output dict → input dict match kare
# # Trick: RunnableSequence automatically pipes output of one into next,
# # so we need small lambda / mapping to keep data structured.

# # Step1: Question → Answer
# first_chain = RunnableSequence(prompt1, model , parser)

# # Step2: Use that answer as "topic" for joke
# second_chain = RunnableSequence(prompt2 , model , parser)

# # Full chain
# #chain = first_chain | (lambda x: {"topic": x}) | second_chain
# chain = RunnableSequence(
#     first_chain,
#     RunnableLambda((lambda x: {"topic": x})),
#     second_chain
# )

# # Run
# response = chain.invoke({"question": "kapil sharma"})
# print(response)
