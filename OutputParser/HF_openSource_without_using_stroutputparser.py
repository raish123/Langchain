#importing the models using Langchain Framework
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
#single turn message ko dynamically we can create PromptTemplate class 
#multi turn message ko dynamically we can create ChatPromptTemplate class 
from dotenv import load_dotenv

#now loading the environment variable to this file.
load_dotenv()

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Model name (free open source)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically uses GPU if available
    torch_dtype="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

# Wrap pipeline in LangChain LLM
model = HuggingFacePipeline(pipeline=pipe)

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


#invoking the first propmpt
first_prompt = prompt1.invoke(
    {
        'topic': 'blackhole'
    }
)


#now passing this dynamic prompt to model
result = model.invoke(first_prompt)


##invoking the second propmpt passing the input to 2nd prompt which is coming from llm 
second_prompt = prompt2.invoke(
    {
        'text': result.content
    }
)



#now passing this second prompt to model to get summary of 5 line
response = model.invoke(second_prompt)

print(response.content)