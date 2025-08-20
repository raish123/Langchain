#importing modules
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.prompts import PromptTemplate


#Now creating an object of ChatGoogleGenerativeAI class.
models = ChatOpenAI(
    model="gpt-3.5-turbo",temperature=0.5
)



#creating simple UI page using Streamlit.
st.header('Research Paper Explainer')


# Now we are providing Suggestions to users to select from dropdown list.
# Dropdowns
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

#kaunse style mei research paper dekhna hai.
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

#how you want to see the output result.
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)



templates_that_passing_llm ="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""



#now creating an object of PromptTemplate class.
prompt = PromptTemplate(
    input_variables=["paper_input","style_input","length_input"],
    template=templates_that_passing_llm,
    validate_template=True
)



#filling the place holder like ["papaer_input","style_input","length_input"] that
#to templates_that_passing_llm and invoking
user_input = prompt.invoke(
    {
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input
    }
)

print(user_input)


#creating button so that once user will click they get reponse based on propmpt.
# Button
if st.button("Summarize"):
    if True:
        st.subheader("Summary")
        response = models.invoke(input=user_input)
        st.write(response.content)
    else:
        st.warning("Please enter some text before summarizing.")