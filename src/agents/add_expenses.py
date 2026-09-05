from typing import Any, Dict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.models.models import IngestRecord, TransactionIngestRequest
from src.util.db import vector_db
from src.util.ingest_transactions import save_and_sync_single_transaction

load_dotenv()

# --- 1. Master Category Definitions ---
ALLOWED_CATEGORIES = [
    "Food & Drinks",
    "Groceries",
    "Shopping",
    "Travel",
    "Transportation",
    "Services",
    "Entertainment",
    "Health",
    "Credit Card",
    "Investments",
    "Rent",
    "Utilities",
    "Telecom",
    "Unknown"
]


def get_categories_prompt_str() -> str:
    """Formats the list of allowed categories for LLM prompt injection."""
    return ", ".join(ALLOWED_CATEGORIES)


def normalize_and_validate_category(category: str | None) -> str | None:
    """
    Validates case-insensitively against ALLOWED_CATEGORIES.
    Returns the properly cased category name or None if invalid.
    """
    if not category:
        return None

    clean_cat = category.strip().lower()
    for valid_cat in ALLOWED_CATEGORIES:
        if clean_cat == valid_cat.lower():
            return valid_cat
    return None


# --- 2. Tool & Model Setup ---
tavily_client = TavilyClient()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for business/merchant details to identify category."""
    return tavily_client.search(query)


def get_add_system_prompt() -> str:
    categories_str = get_categories_prompt_str()
    return (
        f"You are a transaction extraction assistant.\n"
        f"Extract transaction details from the given SMS.\n"
        f"Allowed categories MUST be one of: [{categories_str}].\n"
        f"Payment should be either 'UPI' or 'Card'.\n"
        f"- If the transaction is a personal transfer, unidentifiable person's name, or still ambiguous, set category to 'Unknown'."
    )


def extract_transaction_from_sms(sms_text: str) -> TransactionIngestRequest:
    """Extracts structured JSON conforming to TransactionIngestRequest."""
    extractor = model.with_structured_output(TransactionIngestRequest)
    record = extractor.invoke([
        ("system", get_add_system_prompt()),
        ("user", sms_text)
    ])
    existing_category = find_existing_category_in_db(record.merchant)
    if existing_category:
        record.category = existing_category
    return record


# --- 3. Database Lookup & Persistence ---
def find_existing_category_in_db(merchant: str) -> str | None:
    """Checks vector store for historical merchant categorization."""
    if not merchant:
        return None

    results = vector_db.similarity_search(merchant, k=5)
    for doc in results:
        meta = doc.metadata or {}
        if meta.get("merchant", "").lower() == merchant.lower():
            cat = meta.get("category")
            return normalize_and_validate_category(cat)
    return None


def save_transaction(transaction: TransactionIngestRequest):
    """Persists transaction to CSV and updates Vector DB."""
    save_and_sync_single_transaction(
        transaction.date,
        transaction.merchant,
        transaction.amount,
        transaction.category,
        transaction.payment
    )


# --- 4. Main Controller Function ---
def process_expense_ingest(record: IngestRecord) -> IngestRecord:
    """
    Main orchestration handler.
    Handles extraction, category validation, fallback DB lookup, and user interrupts.
    """
    if not record.message:
        record.is_interrupted = True
        record.message = "SMS input is empty."
        return record

    try:
        # Step 1: Structured extraction
        tx = record.transaction
        if not tx:
            tx = extract_transaction_from_sms(record.message)

        # Validate extracted category against allowed list
        validated_cat = normalize_and_validate_category(tx.category)
        tx.category = validated_cat if validated_cat else "Unknown"
        record.transaction = tx

        # Step 2: Handle 'Unknown' category
        if tx.category == "Unknown":
            # Prompt user with list of valid options
            record.is_interrupted = True
            return record

        # Step 3: Persist transaction
        save_transaction(tx)
        record.is_interrupted = False
        record.message = f"Successfully added ₹{tx.amount:.2f} at {tx.merchant} ({tx.category})."
        return record

    except Exception as e:
        record.is_interrupted = True
        record.message = f"Error processing transaction: {str(e)}"
        return record