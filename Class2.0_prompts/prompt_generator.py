import streamlit as st
from langchain.prompts import PromptTemplate


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



#now saving the prompt template file into Json format
prompt.save("dynamic_template_format.json")