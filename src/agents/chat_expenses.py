import csv
import glob
import os
import sys
import time
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from dateutil import parser
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from src.agents.add_expenses import get_categories_prompt_str
from src.models.models import TransactionRecord
from src.util.db import vector_db
from src.util.ingest_transactions import sync_csv_to_vector_db
from src.util.utilities import get_merchant_name, get_date_list, parse_date, extract_amount_from_query, \
    extract_category_and_payment

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

BASE_ICLOUD_DIR = os.path.expanduser(
    os.getenv("BASE_ICLOUD_DIR", "")
)

MAX_RECORD_FETCH_LIMIT = 10
ALLOWED_SYMBOLS = {"^", "$"}

def get_all_records_from_csv(month_year: str) -> tuple[list[dict], dict, dict, float, str]:
    if not month_year or month_year.strip().lower() in ["current", "this month", "latest", ""]:
        month_year = datetime.now().strftime("%b_%Y")

    clean_month = month_year.replace("-", "_").replace(" ", "_")
    month_map = {
        "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
        "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
        "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"
    }
    for full_name, short_name in month_map.items():
        if full_name in clean_month.lower():
            clean_month = clean_month.lower().replace(full_name, short_name)

    file_path = os.path.join(BASE_ICLOUD_DIR, f"{clean_month}.csv")
    if not os.path.exists(file_path):
        matched = glob.glob(os.path.join(BASE_ICLOUD_DIR, f"*{clean_month[:3]}*.csv"))
        if matched:
            file_path = matched[0]
        else:
            return [], {}, {}, 0, file_path

    records = []
    category_totals = {}
    payment_totals = {}
    total_spent = 0.0

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = (row.get("Date") or row.get("date") or "").strip()
                merchant = (row.get("Merchant") or row.get("merchant") or "Unknown").strip()
                raw_amt = str(row.get("Amount") or row.get("amount") or "0").replace(",", "").strip()
                amount = float(raw_amt)
                category = (row.get("Category") or row.get("category") or "Uncategorized").strip()
                payment = (row.get("Payment") or row.get("PaymentMode") or "UPI").strip()

                records.append({
                    "date": date,
                    "merchant": merchant,
                    "amount": amount,
                    "category": category,
                    "payment": payment
                })

                total_spent += amount
                category_totals[category] = category_totals.get(category, 0.0) + amount
                payment_totals[payment] = payment_totals.get(payment, 0.0) + amount
            except (ValueError, AttributeError):
                continue
    return records, category_totals, payment_totals, total_spent, file_path

@tool
def get_monthly_transaction_summary(
    month_year: str = ""
) -> str:
    """
    Reads the monthly CSV file directly and calculates exact totals, breakdowns, or specific date filters.
    Return a human-readable summary of the monthly transactions, including highest and lowest expenses, category and payment breakdowns, and top 5 highest expenses.
    """
    records, category_totals, payment_totals, total_spent, file_path = get_all_records_from_csv(month_year)
    if not records:
        return f"No transactions recorded for {os.path.basename(file_path)}."

    records_sorted = sorted(records, key=lambda x: x["amount"], reverse=True)
    highest = records_sorted[0]
    lowest = records_sorted[-1]

    category_breakdown = ", ".join([f"{cat}: ₹{amt:,.2f}" for cat, amt in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)])
    payment_breakdown = ", ".join([f"{pm}: ₹{amt:,.2f}" for pm, amt in sorted(payment_totals.items(), key=lambda x: x[1], reverse=True)])
    top_5_txns = "\n".join([f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']})" for r in records_sorted[:5]])

    final_result = (
        f"Overview for {os.path.basename(file_path)}:\n"
        f"EXACT CALCULATED MONTH TOTAL: ₹{total_spent:,.2f} across {len(records)} transactions\n"
        f"Highest Expense: ₹{highest['amount']:,.2f} at {highest['merchant']} on {highest['date']} ({highest['category']})\n"
        f"Lowest Expense: ₹{lowest['amount']:,.2f} at {lowest['merchant']} on {lowest['date']} ({lowest['category']})\n"
        f"Categories: {category_breakdown}\n"
        f"Payment Modes: {payment_breakdown}\n"
        f"Top 5 Highest Expenses:\n{top_5_txns}"
    )
    return final_result

@tool
def global_filter_search(
    month_year: str = "",
    date_filter: str = "",
    category_filter: str = "",
    payment_mode_filter: str = "",
    merchant_filter: str = "",
    amount_filter: str = "",
    last_nth_record_filter: str = "",
) -> str :
    """
    Searches across all CSV files in the iCloud directory for transactions matching the provided filters.
    Returns a summary of matching transactions, including totals and breakdowns.

     MANDATORY USAGE RULES:
    - For questions about TODAY, YESTERDAY, or a SPECIFIC DATE: pass `date_filter="DD-Mon-YYYY"`.
    - For questions about a specific merchant in a month: pass `merchant_filter="<merchant_name>"`.
    - For questions about a category in a month: pass `category_filter="<category_name>"`.
    - For full monthly breakdown: pass `month_year="Aug_2026"`.
    - For amount-based filtering: pass `amount_filter` with a comparison operator (e.g., ">1000", "<500", "=500").
    - For highest/lowest expenses, spending breakdowns, or full monthly metrics: pass `month_year="Aug_2026"`.
    - For last N transactions in a month: pass `last_nth_record_filter="^N"` or `last_nth_record_filter="$N"` (e.g., $5 for last 5 transactions or ^10 for first 10 transactions).

    Args:
        month_year: Month identifier (e.g. 'Aug_2026', 'current')
        date_filter: Specific date in 'DD-Mon-YYYY' format (e.g. '21-Aug-2026')
        category_filter: Optional category name
        payment_mode_filter: Optional payment method (e.g. 'UPI', 'Card')
        merchant_filter: Optional merchant name (e.g. 'Swiggy', 'Zomato')
        amount_filter: Optional amount filter (e.g. '>1000', '<500', '=500')
        last_nth_record_filter: Optional Nth record filter (e.g. '5')
    """
    if not month_year:
        return "Month-year is required for filtering. Please provide a valid month (e.g., 'Aug_2026')."

    all_records,_, _, _, file_path = get_all_records_from_csv(month_year)

    if not all_records:
        return "No transactions found across all CSV files."

    reverse = True
    last_nth_record = 0
    if last_nth_record_filter and last_nth_record_filter[0] in ALLOWED_SYMBOLS:
        last_nth_record = int(last_nth_record_filter[1:])
        symbol = last_nth_record_filter[0]
        reverse = {"$": True, "^": False}.get(symbol, True)
    elif last_nth_record_filter and last_nth_record_filter[0].isdigit():
        last_nth_record = int(last_nth_record_filter[0:])
        reverse = True

    # Apply filters
    filtered_records = all_records
    if date_filter:
        clean_date = date_filter.strip().lower()
        filtered_records = [r for r in filtered_records if clean_date in r["date"].lower()]
    if category_filter:
        filtered_records = [r for r in filtered_records if category_filter.lower() in r["category"].lower()]
    if merchant_filter:
        result = get_merchant_name(merchant_filter)
        merchant_filter = result.get("merchant") if result else merchant_filter
        filtered_records = [r for r in filtered_records if merchant_filter.lower() in r["merchant"].lower()]
    if payment_mode_filter:
        clean_filter = payment_mode_filter.lower().replace("credit", "").replace("debit", "").strip()
        filtered_records = [r for r in filtered_records if clean_filter in r["payment"].lower() or payment_mode_filter.lower() in r["payment"].lower()]
    if amount_filter:
        filtered_records = [
            r for r in filtered_records if (
                    (amount_filter.startswith(">") and r["amount"] > float(amount_filter[1:])) or
                    (amount_filter.startswith("<") and r["amount"] < float(amount_filter[1:])) or
                    (amount_filter.startswith("=") and r["amount"] == float(amount_filter))
            )
        ]
    total_records = len(filtered_records)
    filtered_sum = sum(r["amount"] for r in filtered_records[:last_nth_record])

    if last_nth_record_filter:
        max_record = min(int(last_nth_record), MAX_RECORD_FETCH_LIMIT)
        filtered_records = sorted(filtered_records, key=lambda x: parser.parse(x["date"]), reverse=reverse)[:max_record]
    else:
        filtered_records = sorted(filtered_records, key=lambda x: parser.parse(x["date"]), reverse=reverse)
    amount_sum = sum(r["amount"] for r in filtered_records)

    filter_parts = []
    if date_filter:
        filter_parts.append(f"Date Filter: {date_filter}")
    if merchant_filter:
        filter_parts.append(f"Merchant Filter: {merchant_filter}")
    if category_filter:
        filter_parts.append(f"Category Filter: {category_filter}")
    if payment_mode_filter:
        filter_parts.append(f"Payment Mode Filter: {payment_mode_filter}")
    if amount_filter:
        filter_parts.append(f"Amount Filter: {amount_filter}")

    filters_str = ", ".join(filter_parts)
    direction = {True: "last", False: "first"}.get(reverse, "\b")
    if filters_str:
        filters_str = "for filters - " + filters_str

    if 0 < last_nth_record <= MAX_RECORD_FETCH_LIMIT:
        limit_msg = ""
        transactions_msg = f"Here are the {direction} {last_nth_record} transactions from {os.path.basename(file_path)}:\n" + \
                            "\n".join([f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in filtered_records])
    elif last_nth_record > MAX_RECORD_FETCH_LIMIT:
        limit_msg = f"Maximum {MAX_RECORD_FETCH_LIMIT} are records allowed for display\n"
        transactions_msg = f"Here are the {direction} {min(len(filtered_records), MAX_RECORD_FETCH_LIMIT)} transactions from {os.path.basename(file_path)}:\n" + \
                            "\n".join([f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in filtered_records[:MAX_RECORD_FETCH_LIMIT]])
    else:
        limit_msg = ""
        transactions_msg = f"Here are the {direction} {min(len(filtered_records), MAX_RECORD_FETCH_LIMIT)} transactions from {os.path.basename(file_path)}:\n" + \
                           "\n".join([f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in filtered_records[:MAX_RECORD_FETCH_LIMIT]])


    return (
        f"Filtered Transactions Summary:\n"
        f"{filters_str}\n"
        f"Total Transactions Found: {total_records}\n"
        f"EXACT CALCULATED TOTAL AMOUNT: ₹{amount_sum:,.2f} for {len(filtered_records)} transactions\n"
        f"EXACT CALCULATED TOTAL AMOUNT FOR REQUESTED {last_nth_record} is ₹{filtered_sum:,.2f}"
        f"{limit_msg}"
        f"{transactions_msg}" if len(filtered_records) > 0 else ""
    )

@tool
def search_transaction_date_range(
        start_date: str,
        end_date: str,
        category: str,
        payment: str,
        merchant: str = None
) -> str:
    """
    Searches for transactions within a specific date range.
    Returns calculated totals, category breakdowns, and a compact transaction list.
    """
    date_range_list = get_date_list(start_date, end_date)
    if not date_range_list:
        return f"Invalid date range: {start_date} to {end_date}. Please provide valid dates in DD-Mon-YYYY format."

    filter_date = {
        "$or": [{"date": d} for d in date_range_list]
    }

    if category:
        filter_date["category"] = category # type: ignore
    if payment:
        filter_date["payment"] = payment # type: ignore

    try:
        results = vector_db.similarity_search(
            query = f"{merchant}" if merchant else "transaction expense spend",
            filter=filter_date, # type: ignore
            k=100
        )
    except Exception as e:
        return f"Error querying vector database for date range: {e}"

    # Filter to exact dates in range & sort newest first
    valid_docs = [
        doc for doc in results
        if (
                (not merchant and doc.metadata.get("date") in date_range_list) or
                (merchant and doc.metadata.get("merchant", "").strip().lower() == merchant.lower() and doc.metadata.get(
                    "date") in date_range_list)
        )
    ]

    if not valid_docs:
        return f"No transactions found between {start_date} and {end_date}."

    valid_docs = sorted(
        valid_docs,
        key=lambda x: parser.parse(str(x.metadata.get("date", "01-Jan-1970"))),
        reverse=True
    )

    # 1. Deterministic Python Aggregations
    total_amount = 0.0
    category_totals = {}
    merchant_total = 0.0

    for doc in valid_docs:
        meta = doc.metadata or {}
        category = meta.get("category", "Uncategorized")

        raw_amt = str(meta.get("amount", "0")).replace(",", "").strip()
        try:
            amt = float(raw_amt)
        except ValueError:
            amt = 0.0

        total_amount += amt
        category_totals[category] = category_totals.get(category, 0.0) + amt
        merchant_total += amt if merchant and meta.get("merchant", "").strip().lower() == merchant.lower() else 0.0

    # 2. Build Category Summary String
    cat_summary = ", ".join(
        [f"{cat}: ₹{amt:,.2f}" for cat, amt in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)]
    )

    return (
        f"Date Range Summary ({start_date} to {end_date}):\n"
        f"EXACT CALCULATED RANGE TOTAL: ₹{total_amount:,.2f} ({len(valid_docs)} transactions)\n"
        f"Category Breakdown: {cat_summary}"
        + (f"\nMerchant '{merchant}' Total: ₹{merchant_total:,.2f}" if merchant else "")
    )

@tool
def search_transactions(query: str) -> str:
    """
    Search historical transactions across past records for specific line-item lookups.

    USE THIS TOOL ONLY FOR:
    - Finding specific payments by item, merchant, or amount (e.g., 'Find transaction for welding', 'UPI payment for petrol').
    - Locating individual transaction dates/details (e.g., 'When did I buy shoes?').

    DO NOT USE THIS TOOL FOR:
    - Aggregate metrics, total spend calculations, or monthly breakdowns. Use the monthly summary tool instead.
    """
    if not query or query.strip() in ["*", "all", "everything"]:
        query = "transaction expense spend"

    clean_query = query.strip()
    semantic_text = clean_query
    raw_filters = {}

    # 1. Date filter (e.g., "21-Aug-2026", "today")
    date_match, date_val = parse_date(clean_query)
    if date_match and date_val:
        raw_filters["date"] = date_val
        # Strip date from semantic text to avoid vector noise
        semantic_text = re.sub(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b", "", semantic_text, flags=re.IGNORECASE)

    # 2. Amount filter (e.g., "₹500", "140.56")
    amount_match, amount_val = extract_amount_from_query(clean_query)
    if amount_match and amount_val is not None:
        raw_filters["amount"] = float(amount_val)
        # Strip amount and currency symbols from semantic text
        semantic_text = re.sub(r"(?:₹|rs\.?|inr)?\s*\b\d+(?:,\d+)*(?:\.\d{1,2})?\b\s*(?:inr|rs\.?)?", "", semantic_text, flags=re.IGNORECASE)

    # 3. Category & Payment mode filters
    cat_val, pay_val = extract_category_and_payment(clean_query)
    if cat_val:
        raw_filters["category"] = cat_val
    if pay_val:
        raw_filters["payment"] = pay_val

    # Clean residual whitespace
    semantic_text = re.sub(r"\s+", " ", semantic_text).strip()
    if not semantic_text:
        semantic_text = "transaction expense spend"

    # 4. Construct Chroma-compliant filter ($and syntax for multiple conditions)
    chroma_filter = None
    if len(raw_filters) == 1:
        chroma_filter = raw_filters
    elif len(raw_filters) > 1:
        chroma_filter = {"$and": [{k: v} for k, v in raw_filters.items()]}

    results = []

    # 5. Execute filtered similarity search
    if chroma_filter:
        try:
            results = vector_db.similarity_search(
                query=semantic_text,
                k=8,
                filter=chroma_filter
            )
        except Exception as e:
            print(f"Chroma filtered search failed: {e}", flush=True)

    # 6. Fallback to broad semantic similarity search
    if not results:
        results = vector_db.similarity_search(clean_query, k=10)

    if not results:
        return "No matching transactions found in database."

    records = [f"- {doc.page_content.strip()}" for doc in results]
    return "Retrieved Records:\n" + "\n".join(records)

def parse_transaction(metadata):
    if not metadata:
        return None
    return {
        "date": metadata['date'],
        "merchant": metadata['merchant'],
        "amount": metadata['amount'],
        "category": metadata['category'],
        "payment": metadata['payment'],
        "source_file": metadata['source_file'],
    }

@tool
def search_transactions_to_delete(query: str) -> list[dict] | None:
    """
    Finds and prepares candidate transactions for DELETION.
    STRICT CONSTRAINT:
    - Call this tool ONLY and EXCLUSIVELY if the user explicitly says 'delete', 'remove', 'cancel', or 'erase'.
    - DO NOT call this tool for normal lookups or searches like 'Find me the transaction...' or 'Show me...'
    """
    if not query or query.strip() in ["*", "all", "everything"]:
        query = "transaction expense spend"
    results = vector_db.similarity_search(query, k=5)

    if not results:
        return None

    structured = []
    for doc in results:
        data = parse_transaction(doc.metadata)
        if data:
            record = TransactionRecord(**data)
            structured.append({"id": doc.id, "data": record})
    return structured


checkpointer = MemorySaver()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_system_prompt() -> str:
    now = datetime.now()
    yesterday_dt = now - relativedelta(days=1)
    last_month_dt = now - relativedelta(months=1)

    today_str = now.strftime("%A, %d-%b-%Y")
    today_date_only = now.strftime("%d-%b-%Y")          # "22-Aug-2026"
    yesterday_date_only = yesterday_dt.strftime("%d-%b-%Y") # "21-Aug-2026"
    current_month_str = now.strftime("%b_%Y")           # "Aug_2026"
    last_month_str = last_month_dt.strftime("%b_%Y")    # "Jul_2026"

    return f"""You are an exclusive personal finance and expense tracking assistant.

CURRENT CALENDAR CONTEXT:
- Today's Date: {today_str} ({today_date_only})
- Yesterday's Date: {yesterday_date_only}
- Current Month: {current_month_str}
- Last Month: {last_month_str}

CATEGORY CONTEXT:
- Check the user category and pass the correct category for further processing. If the category is not found, return 'Unknown'.
- {get_categories_prompt_str()}

STRICT SCOPE GUARDRAILS (FINANCE ONLY):
- You are EXCLUSIVELY designed to assist with personal finances, expenses, transactions, budgets, and spending analytics.
- If the user asks ANY non-financial question, politely decline.

CRITICAL TOOL ROUTING RULES:

1. DAILY & RECENT QUERIES (Today, Yesterday, Specific Date):
   - When the user asks "What did I spend today?", "Show today's expenses", or asks about yesterday / a specific date:
     You MUST call `global_filter_search` with BOTH `month_year` and `date_filter`:
     -> For TODAY: `global_filter_search(month_year="{current_month_str}", date_filter="{today_date_only}")`
     -> For YESTERDAY: `global_filter_search(month_year="{current_month_str}", date_filter="{yesterday_date_only}")`
     -> For ANY OTHER DATE: `global_filter_search(month_year="<TargetMonth_YYYY>", date_filter="DD-Mon-YYYY")`
   - NEVER omit `date_filter` when the user asks about a specific day.

2. EXPLICIT MONTH-BOUND QUERIES (User specifically mentions a month or "this month" / "last month"):
   - ONLY call `get_monthly_transaction_summary` when a month is EXPLICITLY mentioned in the user prompt (e.g., "in August", "this month", "last month", "in July").
   - For overall monthly summary: `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>")`
   - For merchant in a specific month: `global_filter_search(month_year="<TargetMonth_YYYY>", merchant_filter="<name>")`
   - For category in a specific month: `global_filter_search(month_year="<TargetMonth_YYYY>", category_filter="<category>")`

3. ADVANCED FILTER BASED QUERIES (User mentions merchant, category, payment mode, or amount in a month):
   - If the user asks for a specific merchant, category, payment mode, or amount in a month (e.g., "How much did I spend on Zomato in August?", "Show me all UPI payments in July", "Find all expenses above 1000 in this month"):
     You MUST call `global_filter_search` with the appropriate filters:
     - For merchant: `global_filter_search(month_year="<TargetMonth_YYYY>", merchant_filter="<merchant_name>")`
     - For category: `global_filter_search(month_year="<TargetMonth_YYYY>", category_filter="<category_name>")`
     - For payment mode: `global_filter_search(month_year="<TargetMonth_YYYY>", payment_mode_filter="<payment_mode>")`
     - For amount: `global_filter_search(month_year="<TargetMonth_YYYY>", amount_filter="<amount_condition>")` (e.g., ">1000", "<500", "=500")
     - If the user asks for last N transactions in a month, you MUST call `global_filter_search(month_year="<TargetMonth_YYYY>", last_nth_record_filter=N)` with the extracted N.
     - User can ask for a specific filter with N (e.g., "Show me last 5 transactions from Zomato in August"), you MUST pass both the merchant and last_nth_record_filter to `global_filter_search`.
   - For queries like "last 5 transactions", "recent 3 expenses". Call `global_filter_search(month_year="<TargetMonth_YYYY>", last_nth_record_filter="^N")` with the extracted N.
   - Use "^N" or first N transactions if the user asks for "first N" and "$N" or last N transactions if the user asks for "last N" or "latest N" or "recent N".
   - If user doesn't mention first or last but just says "N transactions", just pass "N" transactions. Call `global_filter_search(month_year="<TargetMonth_YYYY>", last_nth_record_filter="N")`.
   - If the user does not specify a month, default to the current month.
   - If user mentions specific terms like 'latest', 'recent', 'last', 'first' etc, call global_filter_search with provided filters and month.
   - If user asks for total amount, display the sum along with total number of records for the sum N transactions if mentioned.
   - Maximum N allowed is {MAX_RECORD_FETCH_LIMIT}. If the user asks for more than {MAX_RECORD_FETCH_LIMIT}, return only the last {MAX_RECORD_FETCH_LIMIT} transactions.
   - If user asks more than {MAX_RECORD_FETCH_LIMIT}, respond with: "Maximum {MAX_RECORD_FETCH_LIMIT} are allowed. Here are the last {MAX_RECORD_FETCH_LIMIT} transactions."

4. OPEN-ENDED / ALL-TIME LOOKUPS (NO Month Mentioned in Prompt):
   - MANDATORY: If the user asks about a merchant, item, or category WITHOUT naming a specific month (e.g., "How much did I spend on Blinkit?", "How much on Razorpay?", "Did I spend on movies?", "Find all shoe purchases"):
     DO NOT ASSUME THE CURRENT MONTH. DO NOT call `get_monthly_transaction_summary`.
     You MUST call `search_transactions(query="<merchant_or_item>")`.
   - Pass ONLY the merchant or item keyword (e.g., query="Blinkit", query="Razorpay", query="movies").
   - Respond in natural human-readable text summarizing all-time matches found across all returned dates.

5. SMALLEST/LARGEST PURCHASES:
   - For queries like "smallest purchase", "lowest expense", "biggest purchase", "highest expense" in a month:
     Call `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>")`.

6. DATE RANGE QUERIES:
    - For queries like "between 21-Aug-2026 and 25-Aug-2026" or "What's the expense for last week or last two weeks?":
    - Call `search_transaction_date_range(start_date="DD-Mon-YYYY", end_date="DD-Mon-YYYY")` with the extracted start and end dates.
    - If users also asks for particular merchant within that range, pass as `merchant` parameter to `search_transaction_date_range`.
    - If the user asks for "last week" or "last two weeks", calculate the date range based on the current date and pass it to `search_transaction_date_range`.
    
7. DELETE CASE (Strict JSON Output Contract):
   - Proceed with deletion candidate search ONLY if the user explicitly asks to DELETE, REMOVE, DROP, or ERASE a transaction.
   - For deletion requests: Call `search_transactions_to_delete` with all extracted query filters. For example if user mentions "Delete record from Zomato of 400 paid by Card on 26 Aug 2026", pass the query as "26-Aug-2026 400 Zomato Card"
   - Based on the user query mention all the necessary filters in the query  
   - Filter out all relevant matching transactions from the tool output.
   - If similar matching transactions are found (For example "Delete transaction for 300 to Apple paid by Card" but actual record is paid by UPI), return them in a raw JSON array format with below schema.
   - If no transactions are found for deletion (For example record is totally unrelated or no similarities found), return exactly: "No matching transactions found for deletion."
   - Otherwise, you MUST respond ONLY with a raw, valid JSON array adhering strictly to this schema:
    [
      {{"id": "unique_vector_db_id", "data": {{"date": "...", "merchant": "...", "amount": 0.0, "category": "...", "payment": "...", "source_file": "..."}}}},
      ...
    ]
   - Do NOT include any conversational preamble, markdown explanations, or surrounding text outside of the JSON array.
"""


def summarize_and_trim_history(history_messages, max_raw_messages=6):
    if len(history_messages) <= max_raw_messages:
        return history_messages

    older_messages = history_messages[:-max_raw_messages]
    recent_messages = history_messages[-max_raw_messages:]

    summary_prompt = [
        SystemMessage(
            content="Summarize the following financial conversation concisely into 2-3 key facts, retaining numbers, merchants, and dates."
        ),
        *older_messages,
    ]
    summary_response = model.invoke(summary_prompt)

    summary_system_msg = SystemMessage(
        content=f"Summary of earlier conversation context: {summary_response.content}"
    )

    return [summary_system_msg] + recent_messages

def sanitize_message_history(messages: list) -> list:
    """
    Ensures every AIMessage with tool_calls is followed by all corresponding ToolMessages.
    Explicitly strips both tool_calls and additional_kwargs to prevent OpenAI 400 errors.
    """
    valid_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        # Check for tool_calls in either standard or raw OpenAI kwargs
        has_tool_calls = bool(
            getattr(msg, "tool_calls", None) or
            (isinstance(msg, AIMessage) and msg.additional_kwargs.get("tool_calls"))
        )

        if isinstance(msg, AIMessage) and has_tool_calls:
            # Extract expected IDs from both possible locations
            expected_ids = set()
            if getattr(msg, "tool_calls", None):
                expected_ids.update(tc["id"] for tc in msg.tool_calls if "id" in tc)
            if msg.additional_kwargs.get("tool_calls"):
                expected_ids.update(tc["id"] for tc in msg.additional_kwargs["tool_calls"] if "id" in tc)

            tool_responses = []
            j = i + 1

            # Look ahead for matching ToolMessages
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                if messages[j].tool_call_id in expected_ids:
                    tool_responses.append(messages[j])
                j += 1

            found_ids = {tm.tool_call_id for tm in tool_responses}

            # Only retain if EVERY tool_call_id received a response
            if expected_ids and expected_ids == found_ids:
                valid_messages.append(msg)
                valid_messages.extend(tool_responses)
                i = j  # Jump past the validated tool messages
                continue
            else:
                # Strip tool_calls entirely by wiping additional_kwargs & tool_calls
                clean_kwargs = {k: v for k, v in msg.additional_kwargs.items() if k != "tool_calls"}
                sanitized_ai_msg = AIMessage(
                    content=msg.content or "Checking transactions...",
                    additional_kwargs=clean_kwargs,
                    tool_calls=[]
                )
                valid_messages.append(sanitized_ai_msg)
                i = j
                continue

        # Skip orphan ToolMessages that have no preceding AIMessage
        if isinstance(msg, ToolMessage):
            i += 1
            continue

        valid_messages.append(msg)
        i += 1

    return valid_messages


SESSION_TIMEOUT_SECONDS = 300
session_tracker = {
    "last_seen": 0,
    "active_session_id": "iphone_chat_session_0"
}


def get_effective_thread_id(base_thread_id: str = "iphone_chat_session") -> str:
    now = time.time()
    elapsed = now - session_tracker["last_seen"]

    if elapsed > SESSION_TIMEOUT_SECONDS:
        session_tracker["active_session_id"] = f"{base_thread_id}_{int(now)}"
        print(f"🧹 Context reset! Started new session: {session_tracker['active_session_id']}", flush=True)

    session_tracker["last_seen"] = int(now)
    return session_tracker["active_session_id"]


def query_agent_with_token_limit(user_query: str, thread_id: str = "iphone_chat_session"):
    effective_thread_id = get_effective_thread_id(thread_id)
    config: RunnableConfig = {
        "configurable": {"thread_id": effective_thread_id},
        "recursion_limit": 5
    }

    tools = [global_filter_search, search_transactions, search_transactions_to_delete, get_monthly_transaction_summary, search_transaction_date_range]

    dynamic_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=get_system_prompt(),
        checkpointer=checkpointer,
    )

    # 1. Fetch current checkpointer state
    state = dynamic_agent.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []

    # 2. Trim and Sanitize history
    if len(messages) > 6:
        trimmed_messages = summarize_and_trim_history(messages, max_raw_messages=4)
        clean_messages = sanitize_message_history(trimmed_messages)
        dynamic_agent.update_state(config, {"messages": clean_messages})
    elif messages:
        # Sanitize existing state in case a previous run failed midway
        clean_messages = sanitize_message_history(messages)
        if len(clean_messages) != len(messages):
            dynamic_agent.update_state(config, {"messages": clean_messages})

    # 3. Invoke agent with safe fallback
    try:
        response = dynamic_agent.invoke({"messages": [("user", user_query)]}, config=config) # type: ignore
        messages = response.get("messages", [])
        return messages[-1].content if messages else "No response generated."
    except Exception as e:
        print(f"⚠️ Agent execution error: {e}", flush=True)
        # If checkpointer state got corrupted, reset active session for next prompt
        session_tracker["active_session_id"] = f"{thread_id}_{int(time.time())}"
        raise e


# --- Interactive CLI Mode ---
def start_cli():
    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    cli_thread_id = f"mac_cli_session_{int(time.time())}"

    print(f"{BOLD}{CYAN}=========================================================={RESET}")
    print(f"{BOLD}{CYAN}      💳 Expense Agent — Interactive Mac Terminal         {RESET}")
    print(f"{BOLD}{CYAN}=========================================================={RESET}")
    print(f"{GRAY}Commands: 'exit' to quit | 'clear' to reset screen | 'sync' to pull iCloud{RESET}\n")

    # Initial sync on launch
    if sync_csv_to_vector_db:
        print(f"{YELLOW}🔄 Syncing local vector DB with iCloud CSVs...{RESET}", end="", flush=True)
        try:
            sync_csv_to_vector_db()
            print(f"\r{GREEN}✅ Vector DB is fully synced and ready!{RESET}\n")
        except Exception as e:
            print(f"\r{YELLOW}⚠️ Auto-sync skipped: {e}{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You ➜ {RESET}").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{CYAN}Goodbye! 👋{RESET}")
                break

            if user_input.lower() == "clear":
                os.system("clear")
                continue

            if user_input.lower() == "sync":
                if sync_csv_to_vector_db:
                    print(f"{YELLOW}🔄 Running sync...{RESET}")
                    sync_csv_to_vector_db()
                else:
                    print(f"{YELLOW}⚠️ Sync module not found.{RESET}")
                continue

            if user_input.lower() in ["reset", "new"]:
                cli_thread_id = f"mac_cli_session_{int(time.time())}"
                print(f"{GRAY}🧹 Conversation context cleared. Started fresh session.{RESET}\n")
                continue

            print(f"{GRAY}Thinking...{RESET}", end="\r", flush=True)
            reply = query_agent_with_token_limit(user_input, thread_id=cli_thread_id)

            # Clear the "Thinking..." line and print response
            sys.stdout.write("\033[K")
            print(f"{BOLD}{CYAN}Assistant ➜{RESET}\n{reply}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}Session ended. Goodbye! 👋{RESET}")
            break


if __name__ == "__main__":
    start_cli()