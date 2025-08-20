# 🔹 LangChain Components

LangChain framework consists of several modular components that can be combined to build powerful applications with LLMs:

## 1. Models
Interface for Large Language Models (LLMs) and Embedding Models.  
**Examples:** OpenAI, Hugging Face, Anthropic, Google Gemini.

## 2. Prompts
Templates for structuring user input and dynamic variables into a consistent format for LLMs.  
**Examples:** PromptTemplate, ChatPromptTemplate.

## 3. Chains
Logical sequence of calls combining models, prompts, and other components to create workflows.  
**Examples:** LLMChain, SequentialChain.

## 4. Agents
Decision-making systems that use an LLM to choose and call tools dynamically.  
**Examples:** conversational agents, reasoning agents.

## 5. Memory
Stores and retrieves past interactions to enable contextual and stateful conversations.  
**Examples:** buffer memory, conversation memory.

## 6. Indexes & Retrieval
Manages documents, vector databases, and retrievers for tasks like RAG (Retrieval-Augmented Generation).  
**Examples:** FAISS, Pinecone, Chroma.

## 7. Tools & Integrations
External utilities and APIs that extend LLM capabilities.  
**Examples:** Python REPL, web search, APIs, databases.
