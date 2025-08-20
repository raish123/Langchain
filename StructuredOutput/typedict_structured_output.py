#whatever the inputs we are sending to LLM model in return --->we are getting unstructure plain text information
#what if i can get sturcture  output in format like (json,dict,pydatntic)
#it would be very usefull for certain cases and also DB integration can be done.

#typedict ek issue hai variable kaaa jo dtype define kiya hu.
#user jab input degha typedict usko validate nahi karta like pydantic



from typing import TypedDict,Literal,Optional


#creating a class who inheriting the property from parent class TypedDict 
#with help of them we are defining class variable

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


#creating an object of ChatHuggingFace class
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
)


#defining class variable by using typeddict
class RefundResponse(TypedDict):
    order_id: str
    customer_name: str
    refund_status: str
    estimated_days: int
    extra_notes: str



# Wrap up the LLM for structured output
structured_llm = chat_model.with_structured_output(RefundResponse)

# Create prompt template
prompt = ChatPromptTemplate([
    ("system",
     """
     You are a helpful e-commerce customer support assistant. 
     Always respond in structured JSON dictionary with the following fields:
     order_id, customer_name, refund_status, estimated_days, extra_notes.
     If info is missing, use "Unknown". Do not hallucinate.
     """
    ),
    ("human", "Hi, I am John. My order #12345 was cancelled yesterday. When will I get my refund?")
])

# Formatting prompt below step technique we used to fill the place holder.
result = prompt.invoke({})



response:RefundResponse = structured_llm.invoke(result)
print("Full dict output:\n", response)

