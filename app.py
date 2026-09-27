import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set in the environment variables."
    )


# --------------------------------------------------
# 2. Load OTT content
# --------------------------------------------------

with open("data/content.json", "r", encoding="utf-8") as f:
    ott_content = json.load(f)


# --------------------------------------------------
# 3. Convert OTT records to LangChain Documents
# --------------------------------------------------

documents = []

for item in ott_content:

    document = Document(
        page_content=item["description"],
        metadata={
            "id": item["id"],
            "title": item["title"],
            "genre": item["genre"],
            "language": item["language"],
            "duration_minutes": item["duration_minutes"],
        },
    )

    documents.append(document)


print(f"Documents loaded: {len(documents)}")


# --------------------------------------------------
# 4. Split documents into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# --------------------------------------------------
# 5. Create embedding model
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
)


# --------------------------------------------------
# 6. Create LLM
# --------------------------------------------------

llmClient = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0,
)


# --------------------------------------------------
# 7. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="OTT Content Intelligence API",
    version="1.0.0",
)


# --------------------------------------------------
# 8. Health check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# 9. Temporary direct LLM endpoint
# --------------------------------------------------

@app.get("/ask")
def ask(prompt: str):

    try:

        response = llmClient.invoke(prompt)

        return {
            "response": response.content
        }

    except Exception as e:

        return {
            "error": str(e)
        }