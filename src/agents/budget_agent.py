import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# Inserts project root directory to the start of Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Ingest and Sync modules
from src.util.ingest_transactions import (
    BASE_ICLOUD_DIR,
    DB_DIR,
    sync_csv_to_vector_db,
)

# --- Initialize ChromaDB for Financial Rules & Summaries ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
rules_vector_db = Chroma(
    collection_name="financial_rules",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

# Temporary offline buffer file from Shortcuts
SHORTCUTS_APP_CONTAINER = os.path.expanduser(
    os.getenv("SHORTCUTS_APP_CONTAINER_PATH", "")
)
GENERIC_ICLOUD_CONTAINER = os.path.expanduser(
    os.getenv("GENERIC_ICLOUD_CONTAINER_PATH", "")
)

# --- Default Fallback Limits ---
DEFAULT_BUDGET_RULES = """
User Budget Limits:
- Overall Monthly Total Limit: ₹40,000
- Food & Drinks / Dining Out Limit: ₹5,000
- Shopping / Electronics Limit: ₹10,000
- Automobile / Fuel / Maintenance Limit: ₹2,000
- Entertainment / Movies / Events Limit: ₹3,000
- Services (Rent, Salon, Utilities) Limit: ₹20,000
"""


def get_target_month_file_name(target_date: Optional[datetime] = None) -> str:
    """Returns filename like 'Aug_2026.csv'."""
    dt = target_date or datetime.now()
    return f"{dt.strftime('%b_%Y')}.csv"


def flush_temp_buffer():
    """Flushes any remaining offline records before auditing."""
    buffer_path = SHORTCUTS_APP_CONTAINER if os.path.exists(SHORTCUTS_APP_CONTAINER) else GENERIC_ICLOUD_CONTAINER
    if not os.path.exists(buffer_path):
        return

    try:
        subprocess.run(["/usr/bin/brctl", "download", buffer_path], check=False)
        with open(buffer_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return

        print(f"📦 Flushing offline buffer from {os.path.basename(buffer_path)}...", flush=True)
        lines = content.splitlines()
        reader = csv.reader(lines)
        first_row = True

        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue
            if first_row and row[0].strip().lower() in ["date", "header"]:
                first_row = False
                continue
            first_row = False

            # Appends directly into master CSV
            date_str = row[0].strip() if len(row) > 0 else datetime.now().strftime("%d-%b-%Y")
            category = row[1].strip() if len(row) > 1 else "Uncategorized"
            merchant = row[2].strip() if len(row) > 2 else "Unknown"
            raw_amount = row[3].strip() if len(row) > 3 else "0"
            amount = float(raw_amount.replace(",", ""))
            payment = row[4].strip() if len(row) > 4 else "UPI"

            # Resolve file path based on date
            csv_file = os.path.join(BASE_ICLOUD_DIR, get_target_month_file_name())
            with open(csv_file, mode="a", newline="", encoding="utf-8") as out_f:
                writer = csv.writer(out_f)
                writer.writerow([date_str, category, merchant, f"{amount:.2f}", payment])

        open(buffer_path, "w").close()
        print("✅ Temp buffer merged.", flush=True)
    except Exception as e:
        print(f"⚠️ Error flushing buffer during audit: {e}", flush=True)


# --- Output Schema ---
class BudgetAnalysis(BaseModel):
    month_year: str = Field(
        description="The month and year audited, e.g., 'Aug_2026'."
    )
    total_spent: float = Field(
        description="Total amount spent in the target month."
    )
    is_threshold_breached: bool = Field(
        description="True if any category or overall spend exceeds limits."
    )
    breached_categories: str = Field(
        description="Summary of which categories crossed their limits, or 'None'."
    )
    mom_comparison: str = Field(
        description="Month-over-month comparison insights against past summaries."
    )
    alert_message: str = Field(
        description="A concise, actionable 2-sentence summary/advice for the user."
    )


# --- Tools ---
@tool
def read_monthly_csv(month_year: Optional[str] = None) -> str:
    """Read transactions from the iCloud CSV file for the specified month_year (e.g. 'Aug_2026'). If omitted, reads current month."""
    if not month_year:
        file_name = get_target_month_file_name()
    else:
        file_name = f"{month_year.replace('.csv', '')}.csv"

    current_path = os.path.join(BASE_ICLOUD_DIR, file_name)

    if not os.path.exists(current_path):
        return f"CSV file not found at {current_path}."

    total_spent = 0.0
    category_totals: Dict[str, float] = {}

    with open(current_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                amount_str = row.get("Amount") or row.get("amount") or "0"
                amount = float(amount_str.replace(",", "").strip())
                category = (
                    row.get("Category")
                    or row.get("category")
                    or "Uncategorized"
                ).strip()

                total_spent += amount
                category_totals[category] = category_totals.get(category, 0.0) + amount
            except (ValueError, AttributeError):
                continue

    summary = f"Data for {file_name}:\nTotal Spent: ₹{total_spent:,.2f}\nCategories Breakdown:\n"
    for cat, amt in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        summary += f"- {cat}: ₹{amt:,.2f}\n"

    return summary


@tool
def get_budget_thresholds(query: str = "all spending limits") -> str:
    """Fetch all budget thresholds for categories and monthly limits. Call this tool ONLY ONCE."""
    try:
        results = rules_vector_db.similarity_search(query, k=5)
        if not results:
            return DEFAULT_BUDGET_RULES
        return "Stored Budget Limits:\n" + "\n".join([doc.page_content for doc in results])
    except Exception:
        return DEFAULT_BUDGET_RULES


@tool
def save_monthly_summary(month_year: str, summary_text: str) -> str:
    """Saves the final calculated monthly audit summary into long-term vector memory idempotently."""
    doc_id = f"summary_{month_year.replace('.csv', '')}"
    formatted_text = f"Audit Summary for {month_year}: {summary_text}"

    # Overwrite/upsert summary by ID to avoid duplicate historical summaries
    rules_vector_db.add_texts(
        texts=[formatted_text],
        metadatas=[{"type": "monthly_summary", "month_year": month_year}],
        ids=[doc_id],
    )
    return f"Successfully saved audit summary for {month_year} to Vector DB."


@tool
def get_past_monthly_summaries(query: str = "past monthly summaries") -> str:
    """Searches long-term vector memory for past months' audit summaries. Call this tool ONLY ONCE."""
    try:
        results = rules_vector_db.similarity_search(
            query, k=3, filter={"type": "monthly_summary"}
        )
        if not results:
            return "No past monthly summaries found in memory."
        return "Past Monthly Summaries:\n" + "\n".join([doc.page_content for doc in results])
    except Exception:
        return "No past monthly summaries found in memory."


def send_macos_push_notification(title: str, message: str):
    """Sends a native macOS notification (syncs to paired iPhone/Apple Watch)."""
    clean_title = title.replace('"', '\\"')
    clean_message = message.replace('"', '\\"')
    applescript = f'display notification "{clean_message}" with title "{clean_title}" sound name "Glass"'
    subprocess.run(["osascript", "-e", applescript], check=False)


# --- Agent Setup ---
SYSTEM_PROMPT = """You are an expert personal financial management assistant.

Execute this workflow sequentially without repeating tool calls:
1. Call `read_monthly_csv` ONCE to calculate spending totals for the target month.
2. Call `get_budget_thresholds` ONCE with query 'all spending limits' to fetch user limits.
3. Call `get_past_monthly_summaries` ONCE to fetch previous months' totals.
4. Compare current spending against thresholds and past months.
5. Call `save_monthly_summary` ONCE to record the audit totals in memory.
6. Return the structured output matching `BudgetAnalysis`. Do NOT call any tools after saving the summary.
"""

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=model,
    tools=[
        read_monthly_csv,
        get_budget_thresholds,
        save_monthly_summary,
        get_past_monthly_summaries,
    ],
    system_prompt=SYSTEM_PROMPT,
    response_format=BudgetAnalysis,
)


def run_monthly_audit(target_month_str: Optional[str] = None):
    """
    Main runner function:
    1. Flushes any temp offline records from Shortcuts.
    2. Syncs all CSVs into ChromaDB with deterministic IDs.
    3. Runs the agentic financial audit.
    4. Triggers desktop/iPhone notification.
    """
    print("🚀 [Step 1/3] Flushing offline records & syncing Vector DB...", flush=True)
    flush_temp_buffer()
    sync_csv_to_vector_db()

    target_month = target_month_str or get_target_month_file_name()
    print(f"🔍 [Step 2/3] Running financial audit for {target_month}...", flush=True)

    response = agent.invoke(
        {"messages": [("user", f"Perform a financial audit for {target_month}.")]}
    )
    result: BudgetAnalysis = response.get("structured_response")

    print("\n--- Audit Summary ---")
    print(f"Month: {result.month_year}")
    print(f"Total Spent: ₹{result.total_spent:,.2f}")
    print(f"Threshold Breached: {result.is_threshold_breached}")
    print(f"Breached Categories: {result.breached_categories}")
    print(f"MoM Comparison: {result.mom_comparison}")
    print(f"Alert Message: {result.alert_message}")

    print("\n📣 [Step 3/3] Sending push notification...", flush=True)
    if result.is_threshold_breached:
        send_macos_push_notification(
            title="⚠️ Monthly Budget Alert",
            message=result.alert_message
        )
    else:
        send_macos_push_notification(
            title="✅ Monthly Budget on Track",
            message=f"Total spent: ₹{result.total_spent:,.2f}. All categories within limits."
        )

    return result


if __name__ == "__main__":
    run_monthly_audit()