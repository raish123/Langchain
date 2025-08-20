#importing  gemina module
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

#creating an object of ChatGoogleGenerativeAI class.
chat_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',temperature=0.7
)

#calling the method of ChatGoogleGenerativeAI class to pass the input prompt
response = chat_model.invoke(input="what is the capital state of India")

print(response.content)