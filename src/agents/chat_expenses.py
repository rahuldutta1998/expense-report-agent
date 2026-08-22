import csv
import glob
import os
import sys
import time
import re
from datetime import datetime
from typing import Any, Tuple
from dateutil import parser
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field
from pydantic.v1 import BaseModel

from src.util.ingest_transactions import sync_csv_to_vector_db

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="transaction_vectors",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

BASE_ICLOUD_DIR = os.path.expanduser(
    os.getenv("BASE_ICLOUD_DIR", "")
)

class TransactionRecord(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD or DD-Mon-YYYY", json_schema_extra={"example": "12-Aug-2026"})
    merchant: str = Field(..., json_schema_extra={"example": "Starbucks"})
    amount: float = Field(..., json_schema_extra={"example": 5.50})
    category: str = Field(default="Uncategorized", json_schema_extra={"example": "Food & Dining"})
    payment: str = Field(default="UPI", json_schema_extra={"example": "UPI"})
    source_file: str = Field(default=None, json_schema_extra={"example": "Aug_2026.csv"})

@tool
def get_monthly_transaction_summary(
    month_year: str = "",
    date_filter: str = "",
    category_filter: str = "",
    payment_mode_filter: str = "",
    merchant_filter: str = "",
    amount_filter: str = "",
) -> str:
    """
    Reads the monthly CSV file directly and calculates exact totals, breakdowns, or specific date filters.

    MANDATORY USAGE RULES:
    - For questions about TODAY, YESTERDAY, or a SPECIFIC DATE: pass `date_filter="DD-Mon-YYYY"`.
    - For questions about a specific merchant in a month: pass `merchant_filter="<merchant_name>"`.
    - For questions about a category in a month: pass `category_filter="<category_name>"`.
    - For full monthly breakdown: pass `month_year="Aug_2026"`.
    - For amount-based filtering: pass `amount_filter` with a comparison operator (e.g., ">1000", "<500", "=500").
    - For highest/lowest expenses, spending breakdowns, or full monthly metrics: pass `month_year="Aug_2026"`.

    Args:
        month_year: Month identifier (e.g. 'Aug_2026', 'current')
        date_filter: Specific date in 'DD-Mon-YYYY' format (e.g. '21-Aug-2026')
        category_filter: Optional category name
        payment_mode_filter: Optional payment method (e.g. 'UPI', 'Card')
        merchant_filter: Optional merchant name (e.g. 'Swiggy', 'Zomato')
        amount_filter: Optional amount filter (e.g. '>1000', '<500', '=500')
    """
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
            return f"No CSV record found for {month_year} in Transaction_Record directory."

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

    if not records:
        return f"No transactions recorded for {os.path.basename(file_path)}."

    # 1. Specific Date Filter
    if date_filter:
        clean_date = date_filter.strip().lower()
        filtered_by_date = [r for r in records if clean_date in r["date"].lower()]
        date_sum = sum(r["amount"] for r in filtered_by_date)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} ({r['category']}, {r['payment']})" for r in filtered_by_date]

        if not filtered_by_date:
            return f"No transactions recorded on {date_filter}."

        return (
            f"Transactions on {date_filter} ({os.path.basename(file_path)}):\n"
            f"EXACT CALCULATED DAILY TOTAL: ₹{date_sum:,.2f} ({len(filtered_by_date)} items)\n"
            + "\n".join(lines)
        )

    # 2. Merchant Filter
    if merchant_filter:
        filtered_by_merchant = [r for r in records if merchant_filter.lower() in r["merchant"].lower()]
        merchant_sum = sum(r["amount"] for r in filtered_by_merchant)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in filtered_by_merchant]

        if not filtered_by_merchant:
            return f"No transactions found for '{merchant_filter}' in {os.path.basename(file_path)}."

        return (
            f"Summary for '{merchant_filter}' in {os.path.basename(file_path)}:\n"
            f"EXACT CALCULATED MERCHANT TOTAL: ₹{merchant_sum:,.2f} ({len(filtered_by_merchant)} items)\n"
            + "\n".join(lines)
        )

    # 3. Payment Mode Filter
    if payment_mode_filter:
        clean_filter = payment_mode_filter.lower().replace("credit", "").replace("debit", "").strip()
        filtered_by_pay = [r for r in records if clean_filter in r["payment"].lower() or payment_mode_filter.lower() in r["payment"].lower()]
        pay_sum = sum(r["amount"] for r in filtered_by_pay)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']})" for r in filtered_by_pay]

        if not filtered_by_pay:
            return f"No transactions found for payment mode '{payment_mode_filter}' in {os.path.basename(file_path)}."

        return (
            f"Payment Breakdown for {payment_mode_filter} ({os.path.basename(file_path)}):\n"
            f"EXACT CALCULATED TOTAL: ₹{pay_sum:,.2f} ({len(filtered_by_pay)} items)\n"
            + "\n".join(lines)
        )

    # 4. Category Filter
    if category_filter:
        filtered_by_cat = [r for r in records if category_filter.lower() in r["category"].lower() or category_filter.lower() in r["merchant"].lower()]
        cat_sum = sum(r["amount"] for r in filtered_by_cat)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['payment']})" for r in filtered_by_cat]

        if not filtered_by_cat:
            return f"No transactions found for category '{category_filter}' in {os.path.basename(file_path)}."

        return (
            f"Category Summary for '{category_filter}' in {os.path.basename(file_path)}:\n"
            f"EXACT CALCULATED TOTAL: ₹{cat_sum:,.2f} ({len(filtered_by_cat)} items)\n"
            + "\n".join(lines)
        )

    # 5. Amount Filter
    if amount_filter:
        filtered_by_amount = [
            r for r in records if (
                (amount_filter.startswith(">") and r["amount"] > float(amount_filter[1:])) or
                (amount_filter.startswith("<") and r["amount"] < float(amount_filter[1:])) or
                (amount_filter.startswith("=") and r["amount"] == float(amount_filter))
            )
        ]
        amount_sum = sum(r["amount"] for r in filtered_by_amount)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in filtered_by_amount]

        if not filtered_by_amount:
            return f"No transactions found matching amount filter '{amount_filter}' in {os.path.basename(file_path)}."

        return (
            f"Filtered Amount Summary ({amount_filter}) for {os.path.basename(file_path)}:\n"
            f"EXACT CALCULATED TOTAL: ₹{amount_sum:,.2f}\n"
            f"Transactions ({len(filtered_by_amount)} items):\n" + "\n".join(lines) if lines else "No transactions found for this amount filter."
        )

    # 6. Full Month Overview
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
    "Groceries": ["grocery", "groceries", "supermarket", "blinkit", "zepto", "instamart", "vegetables", "fruits", "milk"],
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

STRICT SCOPE GUARDRAILS (FINANCE ONLY):
- You are EXCLUSIVELY designed to assist with personal finances, expenses, transactions, budgets, and spending analytics.
- If the user asks ANY non-financial question, politely decline.

CRITICAL TOOL ROUTING RULES:

1. DAILY & RECENT QUERIES (Today, Yesterday, Specific Date):
   - When the user asks "What did I spend today?", "Show today's expenses", or asks about yesterday / a specific date:
     You MUST call `get_monthly_transaction_summary` with BOTH `month_year` and `date_filter`:
     -> For TODAY: `get_monthly_transaction_summary(month_year="{current_month_str}", date_filter="{today_date_only}")`
     -> For YESTERDAY: `get_monthly_transaction_summary(month_year="{current_month_str}", date_filter="{yesterday_date_only}")`
     -> For ANY OTHER DATE: `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>", date_filter="DD-Mon-YYYY")`
   - NEVER omit `date_filter` when the user asks about a specific day.

2. EXPLICIT MONTH-BOUND QUERIES (User specifically mentions a month or "this month" / "last month"):
   - ONLY call `get_monthly_transaction_summary` when a month is EXPLICITLY mentioned in the user prompt (e.g., "in August", "this month", "last month", "in July").
   - For merchant in a specific month: `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>", merchant_filter="<name>")`
   - For category in a specific month: `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>", category_filter="<category>")`
   - For overall monthly summary: `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>")`

3. OPEN-ENDED / ALL-TIME LOOKUPS (NO Month Mentioned in Prompt):
   - MANDATORY: If the user asks about a merchant, item, or category WITHOUT naming a specific month (e.g., "How much did I spend on Blinkit?", "How much on Razorpay?", "Did I spend on movies?", "Find all shoe purchases"):
     DO NOT ASSUME THE CURRENT MONTH. DO NOT call `get_monthly_transaction_summary`.
     You MUST call `search_transactions(query="<merchant_or_item>")`.
   - Pass ONLY the merchant or item keyword (e.g., query="Blinkit", query="Razorpay", query="movies").
   - Respond in natural human-readable text summarizing all-time matches found across all returned dates.

4. SMALLEST/LARGEST PURCHASES:
   - For queries like "smallest purchase", "lowest expense", "biggest purchase", "highest expense" in a month:
     Call `get_monthly_transaction_summary(month_year="<TargetMonth_YYYY>")`.

5. DELETE CASE (Strict JSON Output Contract):
   - Proceed with deletion candidate search ONLY if the user explicitly asks to DELETE, REMOVE, DROP, or ERASE a transaction.
   - For deletion requests: Call `search_transactions_to_delete` with all extracted query filters.
   - Filter out all relevant matching transactions from the tool output.
   - If no transactions are found for deletion, return exactly: "No matching transactions found for deletion."
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
    Removes orphaned tool calls that would trigger OpenAI 400 errors.
    """
    valid_messages = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        # Check if message is an AI message requesting tool calls
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            expected_ids = {tc["id"] for tc in msg.tool_calls}
            tool_responses = []

            # Look ahead for matching ToolMessages
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                if messages[j].tool_call_id in expected_ids:
                    tool_responses.append(messages[j])
                j += 1

            # Only keep the tool turn if ALL tool_call_ids have responses
            found_ids = {tm.tool_call_id for tm in tool_responses}
            if expected_ids == found_ids:
                valid_messages.append(msg)
                valid_messages.extend(tool_responses)
                i = j
                continue
            else:
                # Discard the broken tool_calls from the AIMessage or drop it
                msg = AIMessage(content=msg.content or "...")
                valid_messages.append(msg)
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

    tools = [search_transactions, search_transactions_to_delete, get_monthly_transaction_summary]

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
        response = dynamic_agent.invoke({"messages": [("user", user_query)]}, config=config)
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