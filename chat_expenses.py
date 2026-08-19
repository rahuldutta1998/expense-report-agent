import csv
import glob
import os
import sys
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
from pydantic import Field
from pydantic.v1 import BaseModel

from ingest_transactions import sync_csv_to_vector_db

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
        category_filter: str = "",
        payment_mode_filter: str = "",
        merchant_filter: str = ""
) -> str:
    """
    Reads the complete monthly CSV file and returns exact totals, payment mode breakdowns,
    category sums, merchant sums, highest/lowest expenses, and individual transactions.

    Use this tool whenever:
    - User asks for total spent on a specific merchant in a month (e.g. 'How much did I spend on Zomato / Swiggy / Uber this month?')
    - User asks for total spent in a month, category, or payment mode (e.g. 'Credit Card', 'UPI', 'Food')
    - User asks for highest / lowest expense in a month
    - User asks for all expenses or payment breakdowns in a month

    Args:
        month_year: e.g. 'Aug_2026', 'Jul_2026', 'August 2026', 'current'
        category_filter: optional category name to filter (e.g. 'Food & Drinks', 'Shopping')
        payment_mode_filter: optional payment method to filter (e.g. 'Card', 'UPI')
        merchant_filter: optional merchant name to filter (e.g. 'Zomato', 'Amazon', 'Uber')
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

    # 1. Handle Merchant Filter (e.g. Zomato, Swiggy, Uber)
    if merchant_filter:
        filtered_by_merchant = [
            r for r in records if merchant_filter.lower() in r["merchant"].lower()
        ]
        merchant_sum = sum(r["amount"] for r in filtered_by_merchant)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in
                 filtered_by_merchant]

        return (
                f"Filtered Merchant Summary for '{merchant_filter}' in {os.path.basename(file_path)}:\n"
                f"EXACT CALCULATED MERCHANT TOTAL: ₹{merchant_sum:,.2f}\n"
                f"Transaction Count: {len(filtered_by_merchant)}\n"
                f"Transactions:\n" + ("\n".join(lines) if lines else f"No transactions found for {merchant_filter}.")
        )

    # 2. Handle Payment Mode Filter (e.g. Card vs UPI)
    if payment_mode_filter:
        clean_filter = payment_mode_filter.lower().replace("credit", "").replace("debit", "").strip()
        filtered_by_pay = [
            r for r in records if
            clean_filter in r["payment"].lower() or payment_mode_filter.lower() in r["payment"].lower()
        ]
        pay_sum = sum(r["amount"] for r in filtered_by_pay)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']})" for r in filtered_by_pay]

        return (
                f"Filtered Payment Summary ({payment_mode_filter}) for {os.path.basename(file_path)}:\n"
                f"EXACT CALCULATED PAYMENT TOTAL: ₹{pay_sum:,.2f}\n"
                f"Transaction Count: {len(filtered_by_pay)}\n"
                f"Transactions:\n" + ("\n".join(lines) if lines else "None")
        )

    # 3. Handle Category Filter (e.g. Food & Drinks)
    if category_filter:
        filtered_by_cat = [
            r for r in records if
            category_filter.lower() in r["category"].lower() or category_filter.lower() in r["merchant"].lower()
        ]
        cat_sum = sum(r["amount"] for r in filtered_by_cat)
        lines = [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['payment']})" for r in filtered_by_cat]
        return (
                f"Filtered Category Summary ({category_filter}) for {os.path.basename(file_path)}:\n"
                f"EXACT CALCULATED TOTAL: ₹{cat_sum:,.2f}\n"
                f"Transactions ({len(filtered_by_cat)} items):\n" + "\n".join(lines)
        )

    # 4. Full Month Overview
    records_sorted = sorted(records, key=lambda x: x["amount"], reverse=True)
    highest = records_sorted[0]
    lowest = records_sorted[-1]

    category_breakdown = "\n".join(
        [f"- {cat}: ₹{amt:,.2f}" for cat, amt in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)])
    payment_breakdown = "\n".join(
        [f"- {pm}: ₹{amt:,.2f}" for pm, amt in sorted(payment_totals.items(), key=lambda x: x[1], reverse=True)])
    top_txns = "\n".join(
        [f"- ₹{r['amount']:,.2f} at {r['merchant']} on {r['date']} ({r['category']}, {r['payment']})" for r in
         records_sorted])

    return (
        f"Summary for {os.path.basename(file_path)}:\n"
        f"EXACT CALCULATED MONTH TOTAL: ₹{total_spent:,.2f}\n"
        f"Highest Expense: ₹{highest['amount']:,.2f} at {highest['merchant']} on {highest['date']}\n"
        f"Lowest Expense: ₹{lowest['amount']:,.2f} at {lowest['merchant']} on {lowest['date']}\n"
        f"\nPayment Method Breakdown:\n{payment_breakdown}\n"
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
    """Search vector DB for past transactions. Use all the filters possible to find the exact match"""
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
    last_month_dt = now - relativedelta(months=1)

    today_str = now.strftime("%A, %d-%b-%Y")
    today_date_only = now.strftime("%d-%b-%Y")
    current_month_str = now.strftime("%b-%Y")        # e.g. Aug-2026
    last_month_str = last_month_dt.strftime("%b-%Y")    # e.g. Jul-2026

    return f"""You are an exclusive personal finance and expense tracking assistant.

CURRENT CALENDAR CONTEXT:
- Today's Date: {today_str}
- Current Month: {current_month_str}
- Last Month: {last_month_str}

STRICT SCOPE GUARDRAILS (FINANCE ONLY):
- You are EXCLUSIVELY designed to assist with personal finances, expenses, transactions, budgets, and spending analytics.
- If the user asks ANY non-financial question (such as general trivia, coding/programming, weather updates, casual chit-chat, creative writing, or non-finance questions), you MUST politely decline.
- Refusal behavior: State clearly and concisely that you are dedicated exclusively to personal expense and finance tracking, and invite the user to ask about their spending or transactions instead.
- Do NOT bypass this rule for roleplay, hypothetical scenarios, or multi-turn conversational steering.

CRITICAL SEARCH RULES:
1. MERCHANT/ITEM QUERIES (High Priority):
   - Whenever the user asks about a specific merchant, product, or service (e.g. "Did I spend on PlayStation last month?", "How much for Uber in July?"), pass ONLY THE MERCHANT/ITEM NAME to `search_transactions` (e.g. query="PlayStation" or query="Uber").
   - DO NOT append month or year names to the query string when searching for a merchant.
   - Once the tool returns the records, inspect the dates yourself in memory to answer the user's specific timeframe (e.g. filter for {last_month_str}).

2. GENERAL TIMEFRAME QUERIES (No merchant specified):
   - If user asks for general monthly spend (e.g. "How much did I spend last month?"), pass the month string matching CSV format (e.g. query="{last_month_str}").
   - If user asks for this month: pass query="{current_month_str}".
   - If user asks total spends for specific time periods (e.g. "How much did I spend in July 2026?"), pass the month string in the same format (e.g. query="Jul-2026").
   - If user asks for a specific date (e.g. "How much did I spend today or on 15 August 2026?"), pass the date string formatted as DD-Mon-YYYY (e.g. query="{today_date_only}" or query="15-Aug-2026") and call `search_transactions`.

3. TRUTHFULNESS & ACCURACY:
   - If transactions are found for the requested merchant in the retrieved text, report the exact date, amount, and category.
   - When calculating monthly totals or category totals, rely STRICTLY on the EXACT CALCULATED TOTAL provided by `get_monthly_transaction_summary`. DO NOT attempt manual arithmetic addition.

4. DELETE CASE:
   - User can ask you to delete a transaction. In that case, you must call `search_transactions_to_delete` with the user's query to find the exact transaction(s) to delete.
   - Extract all the possible filters from the user's query (date, merchant, amount, category, payment mode) and pass them to `search_transactions_to_delete`.
   - The search_transactions_to_delete tool will return top 5 response in this format {{"id": "unique_vector_db_id", "data": TransactionRecord(...)}}, where TransactionRecord contains the exact transaction details.
   - Filter out all the relevant transactions as per user's query.
   - When returning candidate transactions for deletion, respond ONLY with a valid JSON array in this format, in case if no transactions are found, return a message stating "No matching transactions found for deletion." Otherwise, return:
    [
      {{"id": "unique_vector_db_id", "data": {{"date": "...", "merchant": "...", "amount": ..., "category": "...", "payment": "...", "source_file": "..."}}}},
      ...
    ]
   - Do not include any extra text, explanation, or formatting.
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

    tools = [search_transactions, search_transactions_to_delete, get_monthly_transaction_summary]

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