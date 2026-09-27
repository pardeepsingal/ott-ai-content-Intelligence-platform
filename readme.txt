# OTT Content Intelligence Platform

An AI-powered OTT content intelligence platform built using Python, LangChain, OpenAI, FAISS, and FastAPI.

The project demonstrates an end-to-end RAG (Retrieval-Augmented Generation) pipeline for semantic search and natural-language question answering over OTT content metadata.

---

## 🚀 Project Overview

OTT platforms have large amounts of content metadata such as:

- Title
- Genre
- Language
- Duration
- Description
- Content type

This project converts OTT metadata into embeddings and stores them in a FAISS vector store.

When a user asks a question, the application:

1. Converts the question into a semantic representation.
2. Retrieves the most relevant content from FAISS.
3. Builds a context using the retrieved documents.
4. Sends the context and question to an LLM.
5. Generates a grounded answer using the retrieved information.

---

## 🏗️ Architecture

### Ingestion Pipeline

```text
OTT JSON
   ↓
LangChain Documents
   ↓
Text Chunking
   ↓
OpenAI Embeddings
   ↓
Vector Representations
   ↓
FAISS Vector Store# OTT Content Intelligence Platform

An AI-powered OTT content intelligence platform built using Python, LangChain, OpenAI, FAISS, and FastAPI.

The project demonstrates an end-to-end RAG (Retrieval-Augmented Generation) pipeline for semantic search and natural-language question answering over OTT content metadata.

---

## 🚀 Project Overview

OTT platforms have large amounts of content metadata such as:

- Title
- Genre
- Language
- Duration
- Description
- Content type

This project converts OTT metadata into embeddings and stores them in a FAISS vector store.

When a user asks a question, the application:

1. Converts the question into a semantic representation.
2. Retrieves the most relevant content from FAISS.
3. Builds a context using the retrieved documents.
4. Sends the context and question to an LLM.
5. Generates a grounded answer using the retrieved information.

---

## 🏗️ Architecture

### Ingestion Pipeline

```text
OTT JSON
   ↓
LangChain Documents
   ↓
Text Chunking
   ↓
OpenAI Embeddings
   ↓
Vector Representations
   ↓
FAISS Vector Store

## Runtime RAG Pipeline

User Question
      ↓
Retriever
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Documents
      ↓
Context
      ↓
ChatPromptTemplate
      ↓
ChatOpenAI
      ↓
Final Answer
      ↓
FastAPI Response

## 🛠️ Technology Stack
Python
FastAPI
LangChain
OpenAI
OpenAI Embeddings
FAISS
Pydantic
Uvicorn
python-dotenv

Project Structure

ott-ai-content-intelligence-platform/
│
├── app.py                  # FastAPI application and RAG API
├── index.py                # Data ingestion and FAISS index creation
├── requirements.txt        # Python dependencies
├── .env                    # API keys and environment variables
├── .gitignore
│
├── data/
│   └── content.json        # Sample OTT content metadata
│
└── faiss_index/
    ├── index.faiss         # FAISS vector index
    └── index.pkl           # Document metadata


## Workflow


              PREPARE
                 │
                 ▼
             OTT JSON
                 │
                 ▼
             Documents
                 │
                 ▼
               Chunks
                 │
                 ▼
             Embeddingss
                 │
                 ▼
            Vector Store
                 │
═════════════════╪══════════════════
                 │
                SEARCH
                 │
                 ▼
          User Question
                 │
                 ▼
             Embedding
                 │
                 ▼
          Vector Search
                 │
                 ▼
       Relevant OTT Chunks
                 │
═════════════════╪══════════════════
                 │
               ANSWER
                 │
                 ▼
          Prompt + Context
                 │
                 ▼
             ChatOpenAI
                 │
                 ▼
              Answer


===============================================================

Document -> Chunk -> Embedding  => Vector Store (FAISS, Milvus, etc.) => retrieval (Vector Search) => LLM (ChatGPT, LLaMA, etc.) => Answer


----------------

RAG Pipeline


Question
   ↓
Retriever
   ↓
Relevant OTT chunks
   ↓
Context
   ↓
Prompt
   ↓
GPT
   ↓
Answer


---------
User Question
     ↓
Retriever
     ↓
FAISS similarity search
     ↓
Top k relevant Documents
     ↓
Context
     ↓
Prompt Template
     ↓
ChatOpenAI
     ↓
Final Answer


uvicorn app:app --reload