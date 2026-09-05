import csv
import json
import os
import asyncio
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.models import IngestRecord, TransactionIngestRequest, ChatResponse, ChatRequest, SyncResponse, \
    DeleteResponse, DeleteRequestBody
from src.agents.add_expenses import process_expense_ingest
from src.agents.chat_expenses import query_agent_with_token_limit
from src.util.ingest_transactions import (
    sync_csv_to_vector_db, restore_vector_db, get_records_by_id, delete_transaction_ids, append_transaction_to_csv
)
from src.util.delete_transaction_data import delete_transactions_from_csv

load_dotenv()

SECRET_API_KEY = os.getenv("API_SECRET_KEY")

limiter = Limiter(key_func=get_remote_address)

# Temporary offline buffer file created by Shortcuts
SHORTCUTS_APP_CONTAINER = os.path.expanduser(
    os.getenv("SHORTCUTS_APP_CONTAINER_PATH", "")
)
GENERIC_ICLOUD_CONTAINER = os.path.expanduser(
    os.getenv("GENERIC_ICLOUD_CONTAINER_PATH", "")
)


def get_temp_buffer_path() -> str:
    if os.path.exists(SHORTCUTS_APP_CONTAINER):
        return SHORTCUTS_APP_CONTAINER
    return GENERIC_ICLOUD_CONTAINER


def process_offline_buffer():
    """Reads temp_transaction_data.csv, appends records to monthly CSVs, and clears the buffer."""
    buffer_path = get_temp_buffer_path()
    if not os.path.exists(buffer_path):
        return

    try:
        with open(buffer_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return

        print(f"📦 Found offline records in {os.path.basename(buffer_path)}. Processing...", flush=True)

        lines = content.splitlines()
        reader = csv.reader(lines)
        first_row = True
        processed_count = 0

        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue

            if first_row and row[0].strip().lower() in ["date", "header"]:
                first_row = False
                continue
            first_row = False

            try:
                date_str = row[0].strip() if len(row) > 0 else datetime.now().strftime("%Y-%m-%d")
                category = row[1].strip() if len(row) > 1 else "Uncategorized"
                merchant = row[2].strip() if len(row) > 2 else "Unknown"
                raw_amount = row[3].strip() if len(row) > 3 else "0"
                amount = float(raw_amount.replace(",", ""))
                payment = row[4].strip() if len(row) > 4 else "UPI"

                append_transaction_to_csv(date_str, merchant, amount, category, payment)
                processed_count += 1
            except Exception as row_err:
                print(f"⚠️ Error parsing offline row '{row}': {row_err}", flush=True)

        # Clear temp file without deleting it
        open(buffer_path, "w").close()
        print(f"✅ Merged {processed_count} offline records to CSV files. Cleared temp file.", flush=True)

    except Exception as e:
        print(f"⚠️ Error processing offline buffer: {e}", flush=True)


async def periodic_sync_loop():
    while True:
        try:
            await asyncio.to_thread(process_offline_buffer)
            await asyncio.to_thread(sync_csv_to_vector_db)
        except Exception as e:
            print(f"❌ [Background] Sync error: {e}", flush=True)
        await asyncio.sleep(1800)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Expense RAG Server...", flush=True)
    if not SECRET_API_KEY:
        print("⚠️ Warning: API_SECRET_KEY is not defined in your .env file!", flush=True)

    sync_task = asyncio.create_task(periodic_sync_loop())
    yield
    print("🛑 Shutting down Expense RAG Server...", flush=True)
    sync_task.cancel()


app = FastAPI(title="Personal Expense Agent API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: str = Header(...)):
    if not SECRET_API_KEY or not secrets.compare_digest(x_api_key, SECRET_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return x_api_key

@app.get("/")
def health_check():
    return {"status": "online", "message": "Expense RAG Server is running."}

@app.post("/add-record")
async def ingest_single_transaction_v2(
        payload: IngestRecord,
        _api_key: str = Security(verify_api_key),
):
    try:
        response = process_expense_ingest(payload)
        return response
    except Exception as e:
        print(e, flush=True)
        raise HTTPException(status_code=400, detail=f"Internal server error: {e}")

@app.post("/ingest")
async def ingest_single_transaction(
    payload: TransactionIngestRequest,
    _api_key: str = Security(verify_api_key)
):
    try:
        # 1. Append to master monthly CSV
        await asyncio.to_thread(
            append_transaction_to_csv,
            payload.date,
            payload.merchant,
            payload.amount,
            payload.category,
            payload.payment
        )

        # 2. Sync vector DB (instant incremental scan with deterministic ID check)
        await asyncio.to_thread(sync_csv_to_vector_db)

        return {"status": "success", "message": f"Successfully logged expense for {payload.merchant}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    _api_key: str = Security(verify_api_key),
):
    clean_query = body.query.strip().replace("“", "\"").replace("”", "\"")
    print(f"📥 FastAPI received clean query: repr({repr(clean_query)})", flush=True)
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        reply_content = query_agent_with_token_limit(
            user_query=clean_query, thread_id=body.thread_id
        )
        is_valid_json, response = is_json(reply_content)
        if is_valid_json:
            return ChatResponse(reply_type="transaction", reply=response)
        else:
            return ChatResponse(reply_type="text", reply=reply_content)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/sync", response_model=SyncResponse)
async def trigger_sync(_api_key: str = Security(verify_api_key)):
    try:
        await asyncio.to_thread(process_offline_buffer)
        await asyncio.to_thread(sync_csv_to_vector_db)
        return SyncResponse(message="Vector DB and CSVs successfully updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")

@app.post("/delete", response_model=DeleteResponse)
async def delete_transactions(
        transaction_records : DeleteRequestBody,
        _api_key: str = Security(verify_api_key)) -> DeleteResponse:

    if not transaction_records:
        return DeleteResponse(message="No transactions provided for deletion.")

    records = transaction_records.transaction_records
    if not records:
        return DeleteResponse(message="No transactions provided for deletion.")

    deleted_ids = []
    data = []
    for record in records:
        if record and record.id and record.data:
            deleted_ids.append(record.id)
            if hasattr(record.data, "model_dump"):
                data.append(record.data.model_dump())
            else:
                data.append(record.data)

    if not deleted_ids:
        return DeleteResponse(message="No valid transaction IDs found for deletion.")

    deleted_backup = get_records_by_id(deleted_ids)
    if not deleted_backup:
        return DeleteResponse(message="No matching transactions found in vector database for the provided IDs.")

    try:
        delete_transaction_ids(deleted_ids)
        if not delete_transactions_from_csv(data):
            raise Exception
    except Exception as e:
        print(f"Error deleting transactions: {e}", flush=True)
        restore_vector_db(deleted_backup)
        return DeleteResponse(message="Unable to delete transactions from vector database.")

    return DeleteResponse(message=f"Successfully deleted {len(deleted_ids)} transactions." if len(deleted_ids) > 1 else "Successfully deleted 1 transaction.")

def is_json(text: str) -> tuple[bool, Any] | bool:
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, text

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)