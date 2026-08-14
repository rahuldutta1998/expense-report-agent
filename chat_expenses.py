import csv
import glob
import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "finance_memory_db")
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="transaction_vectors",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

BASE_ICLOUD_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Transaction_Record"
)


@tool
def get_monthly_transaction_summary(month_year: str, category_filter: str = "") -> str:
    """
    Reads the complete monthly CSV file and returns exact totals, category sums,
    highest/lowest expenses, and individual transactions.

    Use this tool whenever:
    - User asks for total spent in a month or category (e.g., 'How much did I spend on food this month?')
    - User asks for highest / lowest expense in a month
    - User asks for all expenses in a month

    Args:
        month_year: e.g. 'Aug_2026', 'Jul_2026', 'August 2026', 'current'
        category_filter: optional category name to filter (e.g. 'Food & Drinks', 'Shopping', 'Services')
    """
    if not month_year or month_year.strip().lower() in ["current", "this month", "latest"]:
        month_year = datetime.now().strftime("%b_%Y")

    # Match month string to filename
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
            return f"No CSV record found for {month_year}."

    records = []
    category_totals = {}
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
            except (ValueError, AttributeError):
                continue

    if not records:
        return f"No transactions recorded for {os.path.basename(file_path)}."

    # If user asked for a specific category (e.g. Food & Drinks / Dining)
    if category_filter:
        filtered_records = [
            r for r in records if
            category_filter.lower() in r["category"].lower() or category_filter.lower() in r["merchant"].lower()
        ]
        if filtered_records:
            cat_sum = sum(r["amount"] for r in filtered_records)
            lines = [f"- ₹{r['amount']:.2f} at {r['merchant']} on {r['date']}" for r in filtered_records]
            return (
                    f"Filtered Category Summary ({category_filter}) for {os.path.basename(file_path)}:\n"
                    f"EXACT CALCULATED TOTAL: ₹{cat_sum:,.2f}\n"
                    f"Transactions:\n" + "\n".join(lines)
            )

    # General month summary
    records_sorted = sorted(records, key=lambda x: x["amount"], reverse=True)
    highest = records_sorted[0]
    lowest = records_sorted[-1]

    category_breakdown = "\n".join([f"- {cat}: ₹{amt:,.2f}" for cat, amt in category_totals.items()])
    top_txns = "\n".join(
        [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']})" for r in records_sorted])

    return (
        f"Summary for {os.path.basename(file_path)}:\n"
        f"EXACT CALCULATED MONTH TOTAL: ₹{total_spent:,.2f}\n"
        f"Highest Expense: ₹{highest['amount']:,.2f} at {highest['merchant']} on {highest['date']} ({highest['category']})\n"
        f"Lowest Expense: ₹{lowest['amount']:,.2f} at {lowest['merchant']} on {lowest['date']}\n"
        f"\nCategory Breakdown:\n{category_breakdown}\n"
        f"\nAll Transactions (Highest to Lowest):\n{top_txns}"
    )

@tool
def search_transactions(query: str) -> str:
    """
    Search vector DB for past transactions.
    - If searching for a merchant/item (e.g., PlayStation, Starbucks, Petrol): pass ONLY the merchant name as query.
    - If searching for general monthly/daily spend without a merchant: pass the month/date (e.g., 'Aug-2026', 'Jul-2026').
    """
    if not query or query.strip() in ["*", "all", "everything"]:
        query = "transaction expense spend"

    # Retrieve top 40 records to guarantee full coverage
    results = vector_db.similarity_search(query, k=40)

    if not results:
        return "No matching transactions found in database."

    records = [f"- {doc.page_content}" for doc in results]
    return "Retrieved Records:\n" + "\n".join(records)


checkpointer = MemorySaver()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_system_prompt() -> str:
    now = datetime.now()
    last_month_dt = now - relativedelta(months=1)

    today_str = now.strftime("%A, %d-%b-%Y")
    current_month_str = now.strftime("%b-%Y")      # e.g. Aug-2026
    last_month_str = last_month_dt.strftime("%b-%Y")  # e.g. Jul-2026

    return f"""You are an expert personal finance and expense tracking assistant.

CURRENT CALENDAR CONTEXT:
- Today's Date: {today_str}
- Current Month: {current_month_str}
- Last Month: {last_month_str}

CRITICAL SEARCH RULES:
1. MERCHANT/ITEM QUERIES (High Priority):
   - Whenever the user asks about a specific merchant, product, or service (e.g. "Did I spend on PlayStation last month?", "How much for Uber in July?"), pass ONLY THE MERCHANT/ITEM NAME to `search_transactions` (e.g. query="PlayStation" or query="Uber").
   - DO NOT append month or year names to the query string when searching for a merchant.
   - Once the tool returns the records, inspect the dates yourself in memory to answer the user's specific timeframe (e.g. filter for {last_month_str}).

2. GENERAL TIMEFRAME QUERIES (No merchant specified):
   - If user asks for general monthly spend (e.g. "How much did I spend last month?"), pass the month string matching CSV format (e.g. query="{last_month_str}").
   - If user asks for this month: pass query="{current_month_str}".

3. TRUTHFULNESS & ACCURACY:
   - If transactions are found for the requested merchant in the retrieved text, report the exact date, amount, and category.
   - When calculating monthly totals or category totals, rely STRICTLY on the EXACT CALCULATED TOTAL provided by `get_monthly_transaction_summary`. DO NOT attempt manual arithmetic addition.
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
    config: RunnableConfig = {"configurable": {"thread_id": effective_thread_id}}

    tools = [search_transactions, get_monthly_transaction_summary]

    dynamic_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=get_system_prompt(),
        checkpointer=checkpointer,
    )
    state = dynamic_agent.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []

    if len(messages) > 6:
        trimmed_messages = summarize_and_trim_history(messages, max_raw_messages=4)
        dynamic_agent.update_state(config, {"messages": trimmed_messages})

    response = dynamic_agent.invoke({"messages": [("user", user_query)]}, config=config)

    messages = response.get("messages", [])
    return messages[-1].content if messages else "No response generated."