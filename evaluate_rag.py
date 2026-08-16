from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from chat_expenses import search_transactions, get_monthly_transaction_summary, get_system_prompt, checkpointer

# Import your actual agent tools and query runner

load_dotenv()

# Evaluator LLM (uses gpt-4o for unbiased, high-precision judging)
judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

agent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# --- 1. Evaluation Output Schema ---
class RAGEvaluationScore(BaseModel):
    faithfulness_score: float = Field(
        description="Score from 0.0 to 1.0 on whether facts/numbers are grounded strictly in retrieved context."
    )
    relevance_score: float = Field(
        description="Score from 0.0 to 1.0 on whether the response directly answers the user's intent."
    )
    numeric_accuracy_score: float = Field(
        description="Score from 0.0 to 1.0 for exact math, date matching, and currency calculation accuracy."
    )
    retrieval_quality_score: float = Field(
        description="Score from 0.0 to 1.0 on whether the retrieved records contained the necessary data."
    )
    overall_score: float = Field(
        description="Weighted composite score out of 10.0."
    )
    verdict: str = Field(
        description="'PASS' if overall_score >= 8.0, otherwise 'FAIL'."
    )
    reasoning: str = Field(
        description="Detailed breakdown of why points were awarded or deducted."
    )
    improvement_recommendation: Optional[str] = Field(
        description="Specific actionable advice to improve prompt, tool query, or database chunking."
    )


# --- 2. Judge Prompt ---
JUDGE_SYSTEM_PROMPT = """You are an impartial expert evaluating a Retrieval-Augmented Generation (RAG) system for personal expense tracking.

You will be given:
1. User Query
2. Expected Ground Truth (Optional context on what should happen)
3. Assistant Generated Answer
4. Retrieved Transaction Chunks

Evaluation Criteria:
1. **Faithfulness (0.0 - 1.0)**: Did the LLM fabricate any transaction or amount not present in the retrieved records?
2. **Relevance (0.0 - 1.0)**: Does the answer directly answer what was asked without unnecessary excuses?
3. **Numeric Accuracy (0.0 - 1.0)**: Are the sums, dates (e.g. July vs August), and merchants 100% mathematically and factually precise?
4. **Retrieval Quality (0.0 - 1.0)**: Were the relevant transactions actually fetched by the tool?

Score harshly on date mismatches and numerical math errors.
"""

evaluator_chain = judge_llm.with_structured_output(RAGEvaluationScore)


def evaluate_single_turn(
        user_query: str,
        expected_note: str = "",
        test_session_id: str = "eval_session"
):
    print(f"\n🧪 Running Test: '{user_query}'...")

    session_id = f"{test_session_id}_{int(datetime.now().timestamp())}"
    config = {"configurable": {"thread_id": session_id}}

    # Invoke agent
    dynamic_agent = create_agent(
        model=agent_llm,
        tools=[search_transactions, get_monthly_transaction_summary],
        system_prompt=get_system_prompt(),
        checkpointer=checkpointer,
    )

    response = dynamic_agent.invoke({"messages": [("user", user_query)]}, config=config)
    messages = response.get("messages", [])
    response_text = messages[-1].content if messages else "No response generated."

    # Extract all tool messages and retrieved context directly from the execution trace
    tool_outputs = []
    for m in messages:
        if hasattr(m, "tool_call_id") or m.type == "tool":
            tool_outputs.append(f"Tool Output ({getattr(m, 'name', 'tool')}):\n{m.content}")

    context_str = "\n\n".join(tool_outputs) if tool_outputs else "No external tool was invoked."

    # Invoke Judge with actual trace context
    judge_prompt = f"""
[USER QUERY]
{user_query}

[EXPECTED BEHAVIOR / GROUND TRUTH]
{expected_note}

[ACTUAL CONTEXT RETRIEVED / RETURNED BY TOOLS]
{context_str}

[ASSISTANT GENERATED ANSWER]
{response_text}
"""

    eval_result: RAGEvaluationScore = evaluator_chain.invoke([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt}
    ])

    return eval_result, response_text


# --- 3. Benchmark Test Suite ---
BENCHMARK_TESTS = [
    # --- 1. Single Merchant & All-Time Retrieval ---
    {
        "query": "Did I spend anything on PlayStation?",
        "expected": "Must search for 'PlayStation' across all time without restricting to the current month, retrieving the ₹1,199.00 transaction on 23-Jul-2026 under 'Entertainment'."
    },
    {
        "query": "How much have I spent on Zomato in total this month?",
        "expected": "Must identify August 2026 Zomato transactions, filter out July transactions, and return the exact calculated sum."
    },

    # --- 2. Temporal & Date Grounding ---
    {
        "query": "What expenses did I incur yesterday?",
        "expected": "Must resolve 'yesterday' relative to the system date context, fetch records for that specific date only, or state zero expenses if none exist."
    },
    {
        "query": "Did I pay any bills or services in July 2026?",
        "expected": "Must fetch July 2026 records categorized under 'Services' (e.g., Hathway, mobile recharge, rent) and list them with amounts."
    },

    # --- 3. Sorting & Numerical Extremes (Max / Min) ---
    {
        "query": "What was my highest single expense in July 2026?",
        "expected": "Must use get_monthly_transaction_summary to inspect the complete July CSV and return ₹1,660.22 at AMAZON PAY IN E on 21-Jul-2026."
    },
    {
        "query": "What was my smallest purchase in August 2026?",
        "expected": "Must locate the minimum transaction amount recorded in Aug_2026.csv (e.g. chai, small grocery, or UPI micro-payment) without skipping low-value records."
    },

    # --- 4. Category-Level Aggregations ---
    {
        "query": "How much did I spend on Food & Drinks this month?",
        "expected": "Must calculate the exact sum for 'Food & Drinks' / dining in August 2026 using Python math rather than mental LLM arithmetic."
    },
    {
        "query": "Give me a category-wise breakdown of all expenses for July 2026.",
        "expected": "Must output distinct category sums (e.g. Shopping, Services, Food & Drinks, Entertainment) matching July's total spend."
    },

    # --- 5. Multi-Condition & Payment Mode Filtering ---
    {
        "query": "How much did I spend using my Credit Card in July 2026?",
        "expected": "Must filter transactions specifically where payment mode contains 'Card' or 'Credit Card' and calculate the accurate sum."
    },
    {
        "query": "List all transactions above ₹1,000 across July and August 2026.",
        "expected": "Must retrieve only records where amount >= 1000.00 and list them with their merchant, date, and exact figure."
    },

    # --- 6. Anti-Hallucination & Negative Testing (Zero Records) ---
    {
        "query": "Did I buy anything from Apple or Nike recently?",
        "expected": "If no transactions exist for 'Apple' or 'Nike', must explicitly state that zero records were found. Must NOT invent placeholder transactions."
    },
    {
        "query": "How much did I spend on flights or flight tickets in August 2026?",
        "expected": "Must confirm that no flight/airline expenses are logged in August 2026."
    },

    # --- 7. Scope & Guardrails ---
    {
        "query": "Write a Python script to sort a binary search tree.",
        "expected": "Must politely decline to write generic code and reaffirm its dedicated role as a personal expense assistant."
    },
    {
        "query": "What is the capital of France?",
        "expected": "Must decline answering general trivia and guide the user back to personal finance/expense queries."
    }
]


def run_benchmark_suite():
    print("==================================================")
    print("📊 STARTING RAG EVALUATION BENCHMARK SUITE")
    print("==================================================")

    total_score = 0.0
    passed_tests = 0

    for idx, test in enumerate(BENCHMARK_TESTS, 1):
        score, response = evaluate_single_turn(
            user_query=test["query"],
            expected_note=test["expected"]
        )

        total_score += score.overall_score
        if score.verdict == "PASS":
            passed_tests += 1

        print(f"\n--- [Test {idx}/{len(BENCHMARK_TESTS)}] Result: {score.verdict} ({score.overall_score}/10) ---")
        print(f"🤖 Output: {response}")
        print(f"📈 Faithfulness: {score.faithfulness_score:.2f} | Relevance: {score.relevance_score:.2f} | Numeric: {score.numeric_accuracy_score:.2f} | Retrieval: {score.retrieval_quality_score:.2f}")
        print(f"📝 Reasoning: {score.reasoning}")
        if score.improvement_recommendation:
            print(f"💡 Suggestion: {score.improvement_recommendation}")

    avg_score = total_score / len(BENCHMARK_TESTS)
    print("\n==================================================")
    print(f"🏁 EVALUATION COMPLETE: {passed_tests}/{len(BENCHMARK_TESTS)} Passed | Average Score: {avg_score:.2f}/10.0")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark_suite()