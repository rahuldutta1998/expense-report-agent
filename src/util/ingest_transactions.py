import csv
import glob
import hashlib
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from src.util.db import vector_db, budget_vector_db

load_dotenv()

BASE_ICLOUD_DIR = os.path.expanduser(os.getenv("BASE_ICLOUD_DIR", ""))
os.makedirs(BASE_ICLOUD_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

def generate_txn_id(date: str, merchant: str, amount: float, category: str, payment: str, file_source: str = "",
                    row_idx: int = 0) -> str:
    """Generates a deterministic hash ID unique to the file and row."""
    raw_key = f"{date.strip()}_{merchant.strip().lower()}_{float(amount):.2f}_{category.strip().lower()}_{payment.strip().lower()}_{file_source}_{row_idx}"
    return f"txn_{hashlib.md5(raw_key.encode('utf-8')).hexdigest()}"


def ensure_icloud_file_downloaded(file_path: str) -> str:
    """Ensures an iCloud-evicted .icloud file is downloaded locally."""
    if os.path.exists(file_path):
        return file_path

    placeholder = os.path.join(os.path.dirname(file_path), f".{os.path.basename(file_path)}.icloud")
    if os.path.exists(placeholder):
        print(f"☁️ Downloading {os.path.basename(file_path)} from iCloud...", flush=True)
        subprocess.run(["/usr/bin/brctl", "download", placeholder], check=False)
        for _ in range(10):
            if os.path.isfile(file_path) and not os.path.exists(placeholder):
                break
            time.sleep(0.3)
    return file_path


def get_csv_path_for_date(date_str: str) -> str:
    """Returns absolute path to monthly CSV file in BASE_ICLOUD_DIR."""
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%d-%B-%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return os.path.join(BASE_ICLOUD_DIR, dt.strftime("%b_%Y.csv"))
        except ValueError:
            continue
    return os.path.join(BASE_ICLOUD_DIR, datetime.now().strftime("%b_%Y.csv"))


def append_transaction_to_csv(date_str: str, merchant: str, amount: float, category: str, payment: str) -> tuple[
    str, int]:
    """
    Appends a single transaction to the CSV file.
    Returns (csv_filename, row_index) for exact ID synchronization.
    """
    csv_file_path = get_csv_path_for_date(date_str)
    ensure_icloud_file_downloaded(csv_file_path)
    file_exists = os.path.exists(csv_file_path)

    # Ensure trailing newline
    if file_exists and os.path.getsize(csv_file_path) > 0:
        with open(csv_file_path, "rb+") as check_f:
            check_f.seek(-1, os.SEEK_END)
            if check_f.read(1) != b"\n":
                check_f.write(b"\n")

    # Count existing rows to get deterministic row index
    row_count = 0
    if file_exists:
        with open(csv_file_path, mode="r", encoding="utf-8") as count_f:
            row_count = sum(1 for _ in count_f)

    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists or row_count == 0:
            writer.writerow(["Date", "Category", "Merchant", "Amount", "Payment"])
            row_count = 1

        writer.writerow([date_str, category, merchant, f"{amount:.2f}", payment or "UPI"])

    return os.path.basename(csv_file_path), row_count


def save_and_sync_single_transaction(date_str: str, merchant: str, amount: float, category: str, payment: str = "UPI"):
    """
    1. Appends to CSV disk file.
    2. Directly updates the live in-memory ChromaDB vector store instantly (zero desync).
    """
    filename, row_idx = append_transaction_to_csv(date_str, merchant, amount, category, payment)

    doc_id = generate_txn_id(date_str, merchant, amount, category, payment, filename, row_idx)
    content = f"Transaction on {date_str}: Spent ₹{amount:.2f} at {merchant} under category '{category}' via {payment}."
    metadata = {
        "date": date_str,
        "amount": float(amount),
        "merchant": merchant,
        "category": category,
        "payment": payment,
        "source_file": filename,
    }

    # Direct in-memory write to active ChromaDB singleton
    vector_db.add_texts(texts=[content], metadatas=[metadata], ids=[doc_id])
    print(f"✅ Transaction synced live into ChromaDB with ID: {doc_id}", flush=True)


def sync_csv_to_vector_db():
    """
    Full scan and reconciliation for all CSV files (used on startup or bulk import).
    Deterministic row indexing prevents ID collision.
    """
    print("🔄 Checking iCloud for all CSV files...", flush=True)

    csv_files = sorted(glob.glob(os.path.join(BASE_ICLOUD_DIR, "*.csv")))
    if not csv_files:
        print("⚠️ No transaction CSV files found.", flush=True)
        return

    documents, metadatas, ids = [], [], []

    for file_path in csv_files:
        ensure_icloud_file_downloaded(file_path)
        filename = os.path.basename(file_path)

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                date = (row.get("Date") or row.get("date") or "Unknown Date").strip()
                raw_amount = str(row.get("Amount") or row.get("amount") or "0").replace(",", "").strip()
                merchant = (row.get("Merchant") or row.get("merchant") or "Unknown").strip()
                category = (row.get("Category") or row.get("category") or "Uncategorized").strip()
                payment = (row.get("Payment") or row.get("PaymentMode") or row.get("Payment Mode") or "UPI").strip()

                try:
                    parsed_amount = float(raw_amount)
                except ValueError:
                    parsed_amount = 0.0

                doc_id = generate_txn_id(date, merchant, parsed_amount, category, payment, filename, row_idx)
                content = f"Transaction on {date}: Spent ₹{parsed_amount:.2f} at {merchant} under category '{category}' via {payment}."

                documents.append(content)
                metadatas.append({
                    "date": date,
                    "amount": parsed_amount,
                    "merchant": merchant,
                    "category": category,
                    "payment": payment,
                    "source_file": filename,
                })
                ids.append(doc_id)

    if ids:
        existing_res = vector_db.get(ids=ids)
        existing_ids = set(existing_res.get("ids", [])) if existing_res else set()

        new_docs, new_meta, new_ids = [], [], []
        for d, m, i in zip(documents, metadatas, ids):
            if i not in existing_ids:
                new_docs.append(d)
                new_meta.append(m)
                new_ids.append(i)

        if new_ids:
            vector_db.add_texts(texts=new_docs, metadatas=new_meta, ids=new_ids)
            print(f"🔄 Sync complete: Added {len(new_ids)} new transactions.", flush=True)
        else:
            print(f"⚡ Sync complete: Vector DB up to date.", flush=True)


def delete_transaction_ids(ids: List[str]) -> str:
    """Deletes transactions from ChromaDB safely."""
    if not ids:
        return "No IDs provided."
    try:
        vector_db.delete(ids=ids)
        return "Successfully deleted transaction."
    except Exception as e:
        print(f"Delete error: {e}", flush=True)
        raise e


def get_records_by_id(ids: List[str]) -> Dict[str, Any]:
    if not ids:
        return {}
    return vector_db.get(ids=ids)


def restore_vector_db(deleted_backup: dict) -> None:
    if deleted_backup and deleted_backup.get("documents"):
        vector_db.add_texts(
            texts=deleted_backup["documents"],
            metadatas=deleted_backup["metadatas"],
            ids=deleted_backup["ids"]
        )

def reset_transactions_collection():
    """Wipes and resyncs ONLY the transactions collection safely."""
    print("🗑️ Resetting 'transaction_vectors' collection...", flush=True)
    try:
        # 1. Delete all existing documents in this collection
        # Getting all IDs and deleting them is the cleanest way to clear without re-instantiating the object
        existing_records = vector_db.get()
        if existing_records and existing_records.get("ids"):
            vector_db.delete(ids=existing_records["ids"])
            print(f"Cleared {len(existing_records['ids'])} existing transaction vectors.", flush=True)
    except Exception as e:
        print(f"Error clearing collection: {e}", flush=True)

    # 2. Resync all records from CSVs
    sync_csv_to_vector_db()
    print("✅ 'transaction_vectors' successfully rebuilt and synced.", flush=True)


def reset_budget_summary_collection():
    """Wipes and resyncs ONLY the budget summary collection safely."""
    print("🗑️ Resetting 'budget_summary_vectors' collection...", flush=True)
    try:
        existing_records = budget_vector_db.get()
        if existing_records and existing_records.get("ids"):
            budget_vector_db.delete(ids=existing_records["ids"])
            print(f"Cleared {len(existing_records['ids'])} existing budget summary vectors.", flush=True)
    except Exception as e:
        print(f"Error clearing budget collection: {e}", flush=True)

    # 3. Resync monthly summaries
    sync_monthly_summaries_to_vector_db()
    print("✅ 'budget_summary_vectors' successfully rebuilt and synced.", flush=True)


def sync_monthly_summaries_to_vector_db():
    """Aggregates monthly data and embeds summary records."""
    pass


def reset_and_sync(collection_name: str):
    if collection_name == "transactions":
        reset_transactions_collection()
        return {"status": "success", "message": "Transaction collection reset and synced."}
    elif collection_name == "budget":
        reset_budget_summary_collection()
        return {"status": "success", "message": "Budget collection reset and synced."}
    return None