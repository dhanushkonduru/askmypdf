# AskMyPDF

## 🧠 Chat with Your PDFs – RAG-Powered Multi-PDF Q&A Bot
This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDFs and ask natural language questions about their content. It combines semantic vector search with large language models (LLMs) to deliver accurate, context-rich answers.

The system leverages modern AI components such as vector databases, transformer-based embeddings, and LLMs to make document understanding interactive and scalable.

### 📘 What is RAG?
RAG (Retrieval-Augmented Generation) is a powerful approach that enhances language models by feeding them relevant external context (like documents or PDFs) retrieved through vector search.

How it works:-

Retrieval – Retrieve relevant chunks of data using semantic similarity.

Augmentation – Inject that context into the prompt for the language model.

Generation – Generate a human-like answer using the LLM based on the question + retrieved context.

This allows the model to give grounded, fact-based responses beyond its training knowledge.

### 📦 What is a Vector Database?
A Vector Database stores data as high-dimensional vectors instead of plain text. It enables semantic similarity search, allowing queries like:

"Find me the most relevant paragraphs across these PDFs for the question: 'How does insulin affect blood sugar?'"

In this project, we use Qdrant, a high-performance open-source vector database, to:

Store vector embeddings of PDF chunks

Efficiently search for top-matching document pieces during queries

### 🚀 How It Works
1. PDF Upload & Embedding
Users upload PDF files via the Streamlit interface.

Each PDF is:

Loaded and split into small text chunks

Transformed into vector embeddings using SentenceTransformer

Indexed in Qdrant with metadata (source, text)

2. Querying with RAG
A user enters a question in natural language.

The system:

Converts the question into an embedding

Searches for the most semantically relevant text chunks from all selected PDFs

Aggregates the results and builds a prompt

3. LLM-Powered Answering
The retrieved chunks + question are sent to Groq’s LLaMA 3 70B

The LLM generates a detailed and accurate answer using both the prompt and embedded knowledge

4. Answer Display
The response is displayed in a clean UI

The answer includes source attribution (which PDFs contributed to the answer)

Users can interact in real time and upload more documents

### 🧰 Tech Stack
Component	Technology
UI	Streamlit
Embedding Model	all-MiniLM-L6-v2 (via sentence-transformers)
Text Parsing	LangChain (PyPDFLoader, TextSplitter)
Vector Database	Qdrant
Language Model	Groq API (LLaMA 3 70B)
Backend Logic	Python

### 🧩 Key Features
✅ Multi-PDF semantic search

✅ Automatic PDF embedding

✅ LLM-based natural language answers

✅ Source tracking of answers

✅ Dark/light mode UI

✅ Streamlit-native file uploads and chat flow
