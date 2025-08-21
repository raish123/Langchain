from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# Hugging Face endpoint define karo
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=128,
    do_sample=False,
)

# Chat model objects
model1 = ChatHuggingFace(llm=llm, verbose=True)
model2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
model3 = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)

# Prompt templates
prompt1 = PromptTemplate(
    template="You are a helpful assistant. Please give me notes on this {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Please generate quiz-type short Q&A from the following notes: {notes}",
    input_variables=['notes']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes -> {notes}\nQuiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

# Parser
parser = StrOutputParser()

# Notes chain
notes_chain = prompt1 | model1 | parser

# Quiz chain dependent on notes
quiz_chain = {"notes": notes_chain} | prompt2 | model2 | parser

# Parallel chain (notes + quiz)
parallel_chain = RunnableParallel(
    {
        "notes": notes_chain,
        "quiz": quiz_chain
    }
)

# Merge chain
merge_chain = prompt3 | model3 | parser

# Final chain
chain = parallel_chain | merge_chain

# Call the chain
result = chain.invoke(
    {
        'topic': 'machine learning'
    }
)

print(result)


#to visualize the chains.
#chain.get_graph().print_ascii()