from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Query string")
    thread_id: str = Field(default="iphone_chat_session")

class ChatResponse(BaseModel):
    reply_type: str = Field(default="text", description="Type of reply, e.g., 'text', 'transaction'")
    reply: str | list[dict]
    status: str = "success"

class SyncResponse(BaseModel):
    message: str
    status: str = "success"

class DeleteResponse(BaseModel):
    message: str

class TransactionIngestRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD or DD-Mon-YYYY", json_schema_extra={"example": "12-Aug-2026"})
    merchant: str = Field(..., json_schema_extra={"example": "Starbucks"})
    amount: float = Field(..., json_schema_extra={"example": 5.50})
    category: str = Field(default="Uncategorized", json_schema_extra={"example": "Food & Dining"})
    payment: str = Field(default="UPI", json_schema_extra={"example": "UPI"})
    source_file: str = Field(default="", description="Optional source file name for reference", json_schema_extra={"example": "Aug_2026.csv"})

class IngestRecord(BaseModel):
    message: str = Field(description="Input from SMS or other source, e.g., 'INR 250 spent using ICICI Bank Card on 31-Aug-26 on Rahul Mechanic'")
    is_interrupted: bool = Field(default=False, description="Whether the transaction is interrupted or not.")
    transaction: TransactionIngestRequest | None = Field(description="Transaction ingestion request containing date, category, merchant, amount, and payment details.")

class TransactionRecords(BaseModel):
    id: str = Field(..., description="Deterministic transaction ID")
    data: TransactionIngestRequest = Field(..., description="Transaction data with keys: date, category, merchant, amount, payment")

class DeleteRequestBody(BaseModel):
    transaction_records: list[TransactionRecords] = Field(..., description="List of transactions to delete")

class TransactionRecord(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD or DD-Mon-YYYY", json_schema_extra={"example": "12-Aug-2026"})
    merchant: str = Field(..., json_schema_extra={"example": "Starbucks"})
    amount: float = Field(..., json_schema_extra={"example": 5.50})
    category: str = Field(default="Uncategorized", json_schema_extra={"example": "Food & Dining"})
    payment: str = Field(default="UPI", json_schema_extra={"example": "UPI"})
    source_file: str = Field(default=None, json_schema_extra={"example": "Aug_2026.csv"})

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