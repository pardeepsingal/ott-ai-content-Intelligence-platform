import json
import os
import time
from urllib import response
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

## Load the OTT Contents
load_dotenv()

app = FastAPI(title="OTT AI Content Intelligence Platform", version="1.0.0")

## read the content.json file and load it into a list of documents
with open("data/content.json", "r", encoding="utf-8") as f:
    ott_content = json.load(f)
    
    print(f"OTT content loaded: {len(ott_content)} records")
    
    documents = []
for item in ott_content:
    print(f"ID: {item['id']}, Title: {item['title']}, Genre: {item['genre']}, Language: {item['language']}, Duration: {item['duration_minutes']} minutes")
    
## Prepare the documents for embedding    
    content = f"""
    Title: {item["title"]}
    Genre: {", ".join(item["genre"])}
    Language: {item["language"]}
    Duration: {item["duration_minutes"]} minutes
    Description: {item["description"]}
    """
    doc = Document(
        page_content=content.strip(),
        metadata={
            "id": item["id"],
            "title": item["title"],
            "genre": item["genre"],
            "language": item["language"],
            "duration_minutes": item["duration_minutes"],
        },
    )
    documents.append(doc)
print(f"Documents created: {len(documents)}")

## Convert Documents to Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # number of characters in each chunk
    chunk_overlap=50, # number of characters to overlap between chunkss
    length_function=len,  # function to calculate the length of the text
)
chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    content = chunk.page_content.strip()
    content_length = len(content)
    print(f"Chunk {i+1}: Length: {content_length} characters and content is {content}")
    ## print(f"Chunk {i+1}: {chunk.page_content[:50]}...")  # Print first 50 characters of each chunk


## Embedding the Chunks
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

## Vector Store Creation
vectorstore = FAISS.from_documents(chunks, embedding_model)

#load the FAISS index from local disk
vectorDb = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

### Verify the loaded FAISS index
print(f"Loaded FAISS vector store with {vectorDb.index.ntotal} vectors.")


### Ask a query to the FAISS index
query = "Which movie is about two professionals meeting in Jaipur?"

######################### Now LLM Interaction with the retrieved documents #########################

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0)


### build the retriver.
retriever = vectorDb.as_retriever(search_kwargs={"k": 3}) ## whenever i receive a question, find the relevant 3 most relevant document from FAISS index.
## docs = retriever.get_relevant_documents(query)

#####

prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for an OTT streaming platform.

Answer the user's question using ONLY the context provided below.

If the answer is not available in the context,
say:

"I don't have enough information in the OTT content catalog."

Do not make up information.

Context:
{context}

User Question:
{question}

Answer:
""")

# docs = retriever.invoke(query)


# context = "\n\n".join([doc.page_content for doc in docs])

# message = prompt.invoke({"context": context, "question": query})

# response = llm.invoke(message)

# print (f"\n\n--- LLM Response ---\n{response.content}")

@app.post("/ask")
def ask_question(query: str):
    docs = retriever.invoke(query)
    start_time = time.perf_counter()
    context = "\n\n".join([doc.page_content for doc in docs])
    message = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(message)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "question": query,
        "answer": response.content,
        "retrieved_documents": len(docs),
        "latency_ms": latency_ms,
    }
    
    

answer = ask_question(query)

print(f"\n\n--- LLM Response ---\n{answer}")
print(f"Latency: {latency_ms} ms")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
