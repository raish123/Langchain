#importing modules
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.prompts import PromptTemplate


#Now creating an object of ChatGoogleGenerativeAI class.
models = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",temperature=0.5
)



#creating simple UI page using Streamlit.
st.header('Research Paper Analysis')


# Input box for user text
user_input = st.text_input(
    "Enter your text here:",
    placeholder="Paste or type your text..."
)



#creating button so that once user will click they get reponse based on propmpt.
# Button
if st.button("Summarize"):
    if user_input.strip():
        st.subheader("Summary")
        response = models.invoke(input=user_input)
        st.write(response.content)
        #st.write("👉 Your summarized text will appear here...")
    else:
        st.warning("Please enter some text before summarizing.")