import csv
import os
import shutil
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

BASE_ICLOUD_DIR = os.path.expanduser(
     os.getenv("BASE_ICLOUD_DIR", "")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="transaction_vectors",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

def _resolve_csv_filepath(csv_path: str) -> str | None:
    """
    Accepts file name and checks if it exists in BASE_ICLOUD_DIR.
    Returns the absolute path if found, otherwise None.
    """
    target_path = os.path.join(BASE_ICLOUD_DIR, csv_path)
    if os.path.exists(target_path) and os.path.isfile(target_path):
        return target_path

    return None


def _is_exact_match(csv_row: Dict[str, str], target: Dict[str, Any]) -> bool:
    """
    Matches all 5 fields strictly: Date, Category, Merchant, Amount, Payment.
    Handles numeric formatting (e.g., 98 vs 98.00 vs '98,00').
    """
    row_date = (csv_row.get("Date") or "").strip().lower()
    target_date = str(target.get("date", "")).strip().lower()
    if row_date != target_date:
        return False

    row_category = (csv_row.get("Category") or "").strip().lower()
    target_category = str(target.get("category", "")).strip().lower()
    if row_category != target_category:
        return False

    row_merchant = (csv_row.get("Merchant") or "").strip().lower()
    target_merchant = str(target.get("merchant", "")).strip().lower()
    if row_merchant != target_merchant:
        return False

    row_payment = (csv_row.get("Payment") or "").strip().lower()
    target_payment = str(target.get("payment", "")).strip().lower()
    if row_payment != target_payment:
        return False

    try:
        row_amount = float(str(csv_row.get("Amount", "0")).replace(",", "").strip())
        target_amount = float(str(target.get("amount", "0")).replace(",", "").strip())
        if abs(row_amount - target_amount) > 0.001:
            return False
    except (ValueError, TypeError):
        return False

    return True


def delete_transactions_from_csv(transactions_to_delete: List[Dict[str, Any]]) -> bool:
    """
    Deletes the exact transactions from the corresponding monthly CSV files.

    1. Creates .bak snapshots of all affected CSV files.
    2. Filters out matching rows based on all 5 columns (Date, Category, Merchant, Amount, Payment).
    3. Rewrites modified CSV files.
    4. If any error occurs, automatically rolls back all CSV files from snapshots.

    :param transactions_to_delete: List of dicts with keys: ['date', 'category', 'merchant', 'amount', 'payment']
    :return: (success_bool, deleted_count, message)
    """
    if not transactions_to_delete:
        return True

    # Group targets by target file
    file_to_targets: Dict[str, List[Dict[str, Any]]] = {}
    for txn in transactions_to_delete:
        csv_file = _resolve_csv_filepath(txn.get("source_file", ""))
        if not csv_file:
            return False
        file_to_targets.setdefault(csv_file, []).append(txn)

    # 1. Create backups (.bak) of all files involved before making any modifications
    backup_files: Dict[str, str] = {}
    try:
        for csv_path in file_to_targets.keys():
            bak_path = f"{csv_path}.bak"
            shutil.copy2(csv_path, bak_path)
            backup_files[csv_path] = bak_path

        total_deleted = 0

        # 2. Process each file
        for csv_path, targets in file_to_targets.items():
            fieldnames = ["Date", "Category", "Merchant", "Amount", "Payment"]
            remaining_rows = []
            targets_to_match = list(targets)

            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = reader.fieldnames

                for row in reader:
                    matched_idx = -1
                    for idx, target in enumerate(targets_to_match):
                        if _is_exact_match(row, target):
                            matched_idx = idx
                            break

                    if matched_idx != -1:
                        # Drop row and pop target so duplicate transactions delete 1:1
                        targets_to_match.pop(matched_idx)
                        total_deleted += 1
                    else:
                        remaining_rows.append(row)

            # 3. Rewrite CSV with remaining rows
            with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(remaining_rows)

        # 4. Success: Clean up backup snapshots
        for bak_path in backup_files.values():
            if os.path.exists(bak_path):
                os.remove(bak_path)

        return True

    except Exception as e:
        # Rollback: restore all files from .bak
        print(f"Error deleting transactions: {e}", flush=True)
        for original_path, bak_path in backup_files.items():
            if os.path.exists(bak_path):
                try:
                    shutil.move(bak_path, original_path)
                except Exception as rollback_err:
                    raise f"Critical Rollback Error for {original_path}: {rollback_err}"
        return False