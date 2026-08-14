import csv
import glob
import hashlib
import os
import subprocess
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Unified storage directory in iCloud
BASE_ICLOUD_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Transaction_Record"
)
os.makedirs(BASE_ICLOUD_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

# Initialize Chroma
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="transaction_vectors",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)


def generate_txn_id(date: str, merchant: str, amount: float, category: str, payment: str, occurrence: int = 0) -> str:
    """Generates a deterministic hash ID for a transaction."""
    raw_key = f"{date.strip()}_{merchant.strip().lower()}_{float(amount):.2f}_{category.strip().lower()}_{payment.strip().lower()}_{occurrence}"
    return f"txn_{hashlib.md5(raw_key.encode('utf-8')).hexdigest()}"


def download_current_csv():
    """Requests latest iCloud version of current month's CSV if needed."""
    filename = datetime.now().strftime("%b_%Y.csv")
    file_path = os.path.join(BASE_ICLOUD_DIR, filename)

    if not os.path.exists(file_path):
        return None

    placeholder = f"{os.path.dirname(file_path)}/.{os.path.basename(file_path)}.icloud"
    if not os.path.exists(placeholder):
        return file_path

    print(f"☁️ Requesting latest iCloud version of {filename}...", flush=True)
    subprocess.run(["/usr/bin/brctl", "download", file_path], check=False)

    for _ in range(5):
        if os.path.isfile(file_path) and not os.path.exists(placeholder):
            print(f"✅ {filename} is available locally", flush=True)
            return file_path
        time.sleep(0.5)

    return file_path


def sync_csv_to_vector_db():
    """
    Full scan and sync of all CSV files in iCloud folder to Vector DB.
    Guarantees idempotency and prevents duplicate entries.
    """
    print("🔄 Checking iCloud for new CSV files...", flush=True)
    download_current_csv()

    csv_files = glob.glob(os.path.join(BASE_ICLOUD_DIR, "*.csv"))
    if not csv_files:
        print("⚠️ No transaction CSV files found in folder.", flush=True)
        return

    documents = []
    metadatas = []
    ids = []
    seen_counts = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = (row.get("Date") or row.get("date") or "Unknown Date").strip()
                raw_amount = str(row.get("Amount") or row.get("amount") or "0").replace(",", "").strip()
                merchant = (row.get("Merchant") or row.get("merchant") or "Unknown").strip()
                category = (row.get("Category") or row.get("category") or "Uncategorized").strip()
                payment = (row.get("Payment") or row.get("PaymentMode") or row.get("Payment Mode") or "N/A").strip()

                try:
                    parsed_amount = float(raw_amount)
                except ValueError:
                    parsed_amount = 0.0

                # Count occurrences for transactions identical in every field
                key = (date, merchant.lower(), f"{parsed_amount:.2f}", category.lower(), payment.lower())
                occurrence = seen_counts.get(key, 0)
                seen_counts[key] = occurrence + 1

                # Deterministic ID shared across the whole system
                doc_id = generate_txn_id(date, merchant, parsed_amount, category, payment, occurrence)

                content = (
                    f"Transaction on {date}: Spent ₹{parsed_amount:.2f} at {merchant} "
                    f"under category '{category}' via {payment}."
                )

                documents.append(content)
                metadatas.append(
                    {
                        "date": date,
                        "amount": parsed_amount,
                        "merchant": merchant,
                        "category": category,
                        "payment": payment,
                        "source_file": filename,
                    }
                )
                ids.append(doc_id)

    if ids:
        # Check existing IDs in ChromaDB
        existing_res = vector_db.get(ids=ids)
        existing_ids = set(existing_res.get("ids", [])) if existing_res else set()

        new_docs, new_meta, new_ids = [], [], []
        for d, m, i in zip(documents, metadatas, ids):
            if i not in existing_ids:
                new_docs.append(d)
                new_meta.append(m)
                new_ids.append(i)

        now = datetime.now()
        if new_ids:
            vector_db.add_texts(texts=new_docs, metadatas=new_meta, ids=new_ids)
            print(f"🔄 Auto-sync completed at {now}: Added {len(new_ids)} new transactions to Vector DB.", flush=True)
        else:
            print(f"⚡ Auto-sync completed at {now}: Vector DB is up to date (0 duplicates).", flush=True)


def reset_vector_db():
    """Utility function to wipe and rebuild the vector DB from scratch."""
    print("🗑️ Resetting Vector DB collection...", flush=True)
    vector_db.reset_collection()
    sync_csv_to_vector_db()


if __name__ == "__main__":
    sync_csv_to_vector_db()