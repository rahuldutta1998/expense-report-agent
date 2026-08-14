from dotenv import load_dotenv
from langchain_chroma import Chroma
from ingest_transactions import DB_DIR, embeddings

load_dotenv()

# Initialize Chroma Vector Store
vector_store = Chroma(
    collection_name="transaction_vectors",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

# Fetch all records from the collection
data = vector_store.get()
ids = data.get("ids", [])

if ids:
    # Get the last added document ID (e.g., 'expenses.csv_row_15')
    last_id = ids[-1]

    # Delete by ID
    vector_store.delete(ids=[last_id])
    print(f"✅ Successfully deleted record '{last_id}' from ChromaDB.")
else:
    print("⚠️ Vector store is currently empty.")
