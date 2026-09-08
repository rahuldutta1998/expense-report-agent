import json
import os
import sys
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.add_expenses import ALLOWED_CATEGORIES
from src.models.models import ChatRequest, ChatResponse, TransactionIngestRequest, TransactionRecords, IngestRecord
from src.util.db import vector_db
from src.util.delete_transaction_data import delete_transactions_records

load_dotenv()

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💳", layout="wide")

API_URL = "http://localhost:8080/chat"
ADD_RECORD_URL = "http://localhost:8080/add-record"
API_KEY = os.getenv("API_SECRET_KEY", "")
headers = {"Content-Type": "application/json", "x-api-key": f"{API_KEY}"}

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_deletion_candidates" not in st.session_state:
    st.session_state.pending_deletion_candidates = []


def normalize_candidates(raw_candidates: list) -> list[dict]:
    """Flattens nested data dicts or stringified JSON."""
    flattened = []
    for transaction_items in raw_candidates:
        if not isinstance(transaction_items, dict):
            continue
        if "data" in transaction_items:
            data_val = transaction_items["data"]
            if isinstance(data_val, str):
                try:
                    data_val = json.loads(data_val)
                except Exception as exception:
                    print(exception)
                    data_val = {}
            if isinstance(data_val, dict):
                merged = {**data_val}
                if "id" in transaction_items:
                    merged["id"] = transaction_items["id"]
                flattened.append(merged)
            else:
                flattened.append(transaction_items)
        else:
            flattened.append(transaction_items)
    return flattened

@st.dialog("➕ Add New Transaction")
def add_transaction_modal():
    with st.form("manual_tx_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            tx_date = st.date_input("Date", value=datetime.today())
            merchant = st.text_input("Merchant", placeholder="e.g., Swiggy, Shell")
            amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, format="%.2f")
        with col2:
            category = st.selectbox(
                "Category",
                ALLOWED_CATEGORIES
            )
            payment = st.selectbox("Payment Mode", ["UPI", "Card", "Cash"])

        submitted = st.form_submit_button("Save Transaction", type="primary", use_container_width=True)

        if submitted:
            if not merchant.strip():
                st.error("Please enter a merchant name.")
            elif amount <= 0:
                st.error("Amount must be greater than 0.")
            else:
                formatted_date = tx_date.strftime("%d-%b-%Y")
                clean_merchant = merchant.strip()

                # Construct nested TransactionIngestRequest
                txn_payload = TransactionIngestRequest(
                    date=formatted_date,
                    merchant=clean_merchant,
                    amount=float(amount),
                    category=category,
                    payment=payment,
                    source_file=""
                )

                # Formulate a clear descriptive message for the IngestRecord
                human_message = f"Spent ₹{amount:.2f} at {clean_merchant} on {formatted_date} via {payment} under {category}"

                # Construct top-level IngestRecord
                ingest_record = IngestRecord(
                    message=human_message,
                    is_interrupted=False,
                    transaction=txn_payload
                )

                try:
                    add_record_response = requests.post(
                        ADD_RECORD_URL,
                        json=ingest_record.model_dump(),
                        headers=headers,
                        timeout=20
                    )

                    if add_record_response.status_code in (200, 201):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ Successfully added transaction: ₹{amount:.2f} at **{clean_merchant}** ({category}) on {formatted_date}."
                        })
                        st.success("Transaction added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add record ({add_record_response.status_code}): {add_record_response.text}")
                except Exception as exception:
                    st.error(f"Error connecting to backend: {str(exception)}")

tab_chat, tab_manage = st.tabs(["💬 Chat Assistant", "📊 Manage Transactions"])
# Capture chat input at bottom
prompt = st.chat_input("e.g., 'What did I spend on food?' or Delete the ₹420 Swiggy transaction'")

# Chat Assistant History & Actions
with tab_chat:
    col_title, col_btn = st.columns([0.8, 0.2])
    with col_title:
        st.subheader("Personal Finance AI")
    with col_btn:
        if st.button("➕ Add", type="secondary", use_container_width=True):
            add_transaction_modal()

    # 1. Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 2. Interactive Selection Box (when deletion candidates exist)
    if st.session_state.pending_deletion_candidates:
        candidates = normalize_candidates(st.session_state.pending_deletion_candidates)

        with st.container(border=True):
            st.warning("⚠️ **Select the transaction(s) you want to delete:**")

            df_candidates = pd.DataFrame(candidates)
            if "amount" in df_candidates.columns:
                df_candidates["amount"] = pd.to_numeric(df_candidates["amount"], errors="coerce")
                df_candidates["amount"] = df_candidates["amount"].fillna(0.0)

            event = st.dataframe(
                df_candidates,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                column_config={
                    "id": None,
                    "date": st.column_config.TextColumn("Date"),
                    "merchant": st.column_config.TextColumn("Merchant"),
                    "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                    "category": st.column_config.TextColumn("Category"),
                    "payment": st.column_config.TextColumn("Payment Mode"),
                    "source_file": None,
                },
            )

            selected_rows = event.selection.rows
            col_confirm, col_cancel = st.columns([0.3, 0.7])

            with col_confirm:
                if selected_rows:
                    selected_items = df_candidates.iloc[selected_rows].to_dict(orient="records")

                    if st.button(f"🗑️ Confirm Deletion ({len(selected_items)})", type="primary", use_container_width=True):
                        records_to_delete = []
                        for item in selected_items:
                            ingest_data = TransactionIngestRequest(
                                date=str(item.get("date", "")),
                                merchant=str(item.get("merchant", "")),
                                amount=float(item.get("amount", 0.0)),
                                category=str(item.get("category", "Uncategorized")),
                                payment=str(item.get("payment", "UPI")),
                                source_file=str(item.get("source_file", "") or ""),
                            )
                            records_to_delete.append(TransactionRecords(id=str(item.get("id", "")), data=ingest_data))

                        delete_transactions_records(records_to_delete)
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ Deleted {len(records_to_delete)} transaction(s) successfully."})
                        st.session_state.pending_deletion_candidates = []
                        st.rerun()

            with col_cancel:
                if st.button("❌ Cancel"):
                    st.session_state.pending_deletion_candidates = []
                    st.session_state.messages.append({"role": "assistant", "content": "Deletion cancelled."})
                    st.rerun()

    # 3. Process new prompt live with spinner inside tab_chat
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  # type: ignore
                try:
                    request_payload = ChatRequest(query=prompt, thread_id="streamlit_session").model_dump()
                    res = requests.post(API_URL, json=request_payload, headers=headers, timeout=30)

                    if res.status_code == 200:
                        chat_res = ChatResponse.model_validate(res.json())

                        if chat_res.reply_type == "transaction" and isinstance(chat_res.reply, list):
                            st.session_state.pending_deletion_candidates = chat_res.reply
                            reply_text = f"Found {len(chat_res.reply)} matching transaction(s). Please choose below:"
                        else:
                            reply_text = str(chat_res.reply)

                        st.write(reply_text)
                        st.session_state.messages.append({"role": "assistant", "content": reply_text})

                        if chat_res.reply_type == "transaction":
                            st.rerun()
                    else:
                        err_msg = f"⚠️ Server Error ({res.status_code}): {res.text}"
                        st.write(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                except Exception as e:
                    err_msg = f"❌ Error: {str(e)}"
                    st.write(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})


# Direct Vector DB Table View
with tab_manage:
    st.subheader("All Transaction Records")

    def fetch_all_transactions() -> list[dict]:
        data = vector_db.get()
        if not data or not data.get("ids"):
            return []
        records = []
        for doc_id, meta in zip(data["ids"], data.get("metadatas", [])):
            meta_dict = meta or {}
            records.append({
                "id": doc_id,
                "date": meta_dict.get("date", "Unknown"),
                "merchant": meta_dict.get("merchant", "Unknown"),
                "amount": float(meta_dict.get("amount", 0.0)),
                "category": meta_dict.get("category", "Uncategorized"),
                "payment": meta_dict.get("payment", "UPI"),
                "source_file": meta_dict.get("source_file", ""),
            })
        return records

    all_records = fetch_all_transactions()
    if all_records:
        df_all = pd.DataFrame(all_records)
        if "amount" in df_all.columns:
            df_all["amount"] = pd.to_numeric(df_all["amount"], errors="coerce")
            df_all["amount"] = df_all["amount"].fillna(0.0)

        event_all = st.dataframe(
            df_all,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "id": None,
                "date": st.column_config.TextColumn("Date"),
                "merchant": st.column_config.TextColumn("Merchant"),
                "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "category": st.column_config.TextColumn("Category"),
                "payment": st.column_config.TextColumn("Payment Mode"),
                "source_file": None,
            },
        )

        if event_all.selection.rows:
            selected_items = df_all.iloc[event_all.selection.rows].to_dict(orient="records")
            if st.button(f"🗑️ Delete Selected Records ({len(selected_items)})", type="primary"):
                records_to_delete = [
                    TransactionRecords(
                        id=str(item.get("id", "")),
                        data=TransactionIngestRequest(
                            date=str(item.get("date", "")),
                            merchant=str(item.get("merchant", "")),
                            amount=float(item.get("amount", 0.0)),
                            category=str(item.get("category", "Uncategorized")),
                            payment=str(item.get("payment", "UPI")),
                            source_file=str(item.get("source_file", "") or ""),
                        ),
                    )
                    for item in selected_items
                ]
                delete_transactions_records(records_to_delete)
                st.success(f"Successfully deleted {len(records_to_delete)} transaction(s)!")
                st.rerun()
    else:
        st.info("No transaction records found in Vector DB.")