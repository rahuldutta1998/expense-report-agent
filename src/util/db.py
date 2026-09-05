# src/util/db.py
import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../finance_memory_db"))
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single global persistent client
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Collection 1: Granular Transactions
vector_db = Chroma(
    client=chroma_client,
    collection_name="transaction_vectors",
    embedding_function=embeddings,
)

# Collection 2: Monthly Budget / Summaries
budget_vector_db = Chroma(
    client=chroma_client,
    collection_name="budget_summary_vectors",
    embedding_function=embeddings,
)