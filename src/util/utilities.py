import re
from typing import Any, Tuple

from dateutil.relativedelta import relativedelta
from rapidfuzz import fuzz
from dateutil import parser

from src.util.db import vector_db


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

def get_merchant_name(merchant: str) -> dict | None:
    # 1. Fetch top 5 vector matches (Chroma returns closest distance first)
    results = vector_db.similarity_search_with_score(merchant, k=5)

    if not results:
        return None

    query_lower = merchant.lower()

    # ==========================================
    # TIER 1: Prioritise Perfect Exact Matches First
    # ==========================================
    for doc, score in results:
        db_merchant = doc.metadata.get("merchant", "").lower()
        if query_lower == db_merchant:
            return doc.metadata

    # ==========================================
    # TIER 2: High-Confidence Typos / Partial Matches
    # ==========================================
    for doc, score in results:
        db_merchant = doc.metadata.get("merchant", "").lower()
        fuzzy_score = fuzz.ratio(query_lower, db_merchant)

        if fuzzy_score >= 50:
            return doc.metadata

    # ==========================================
    # TIER 3: Raw Vector Distance Fallback
    # ==========================================
    best_doc, best_distance = results[0]
    if best_distance <= 0.45:
        return best_doc.metadata

    return None

def parse_date(date_str: str) -> tuple[bool, str] | tuple[bool, None]:
    """
    Parses a date string in various formats and returns it in DD-Mon-YYYY format.
    Returns None if parsing fails.
    """
    try:
        parsed_date = parser.parse(date_str)
        formatted_date = parsed_date.strftime("%d-%b-%Y")
        return True, formatted_date
    except (ValueError, TypeError, OverflowError):
        return False, None

def extract_amount_from_query(query: str) -> tuple[bool, Any | None] | tuple[bool, None]:
    """
    Extracts an explicit monetary amount from a query string.
    Handles: '140.56', '21-Aug-2026 140.56', '₹504', 'Rs. 1,250.00', '500 INR'.
    Ignores date numbers like '2026', '21', '08'.
    """
    if not query:
        return False, None

    # 1. Mask out date formats (e.g. 21-Aug-2026, 2026-08-21, 21/08/2026) to prevent date digits matching as amounts
    cleaned_query = re.sub(r"\b\d{1,4}[-/][A-Za-z0-9]{2,3}[-/]\d{2,4}\b", "", query)
    cleaned_query = re.sub(r"\b(19\d\d|20\d\d)\b", "", cleaned_query)  # Mask standalone 4-digit years

    # 2. Match currency symbols/prefixes or standalone decimal/integer numbers
    # Group 1 captures the actual numeric value (e.g. 140.56 or 504 or 1,250.00)
    matches = re.finditer(r"(?<![A-Za-z0-9])(?:₹\s*|rs\.?\s*|inr\s*)?(\d+(?:,\d+)*(?:\.\d{1,2})?)(?![A-Za-z0-9])", cleaned_query, re.IGNORECASE)

    amounts = []
    for match in matches:
        raw_val = match.group(1).replace(",", "").strip()
        try:
            val = float(raw_val)
            if val > 0:
                amounts.append(val)
        except ValueError:
            continue

    return (True, amounts[0]) if amounts else (False, None)

CATEGORY_SYNONYMS = {
    "Transport": ["transport", "transportation", "fuel", "petrol", "diesel", "cab", "uber", "ola", "auto", "metro", "parking", "toll"],
    "Food & Drinks": ["food", "drinks", "dining", "restaurant", "cafe", "coffee", "lunch", "dinner", "breakfast", "swiggy", "zomato"],
    "Groceries": ["grocery", "groceries", "supermarket", "blinkit", "zepto", "Swiggy Insta", "instamart", "vegetables", "fruits", "milk"],
    "Shopping": ["shopping", "clothes", "apparel", "amazon", "flipkart", "electronics", "shoes"],
    "Bills & Utilities": ["bills", "utilities", "electricity", "water", "wifi", "broadband", "mobile", "recharge", "rent", "maintenance"],
    "Entertainment": ["entertainment", "movies", "cinema", "bookmyshow", "games", "playstation", "steam", "netflix", "spotify", "prime"],
}

def resolve_category(query: str) -> str | None:
    query_lower = query.lower()
    for official_cat, keywords in CATEGORY_SYNONYMS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                return official_cat
    return None

KNOWN_PAYMENT_MODES = ["UPI", "Card", "Cash"]


def extract_category_and_payment(query: str) -> Tuple[str | None, str | None]:
    pay_match = None

    cat_match = resolve_category(query)

    for pay in KNOWN_PAYMENT_MODES:
        if re.search(rf"\b{re.escape(pay)}\b", query, re.IGNORECASE):
            pay_match = "Card" if "card" in pay.lower() else pay.upper()
            break

    return cat_match, pay_match

def get_date_list(start_date: str, end_date: str) -> list[str]:
    """
    Generates a list of dates in DD-Mon-YYYY format between start_date and end_date (inclusive).
    """
    try:
        start_dt = parser.parse(start_date)
        end_dt = parser.parse(end_date)
    except (ValueError, TypeError):
        return []

    if start_dt > end_dt:
        return []

    date_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%d-%b-%Y"))
        current_dt += relativedelta(days=1)

    return date_list