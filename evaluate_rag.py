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
    {
        "query": "Did I spend anything on PlayStation?",
        "expected": "Should retrieve the ₹1,199.00 transaction on 23-Jul-2026 under Entertainment without hallucinating August limits."
    },
    {
        "query": "How much did I spend on food and dining this month?",
        "expected": "Should filter transactions for current month and sum up Zomato/Swiggy/Blinkit/dining/etc (anything related to food) accurately."
    },
    {
        "query": "What was my highest single expense in July 2026?",
        "expected": "Should look at July transactions and identify the maximum amount. Actual Record (21-Jul-26	Shopping AMAZON PAY IN E 1660.22 Card)"
    },
    {
        "query": "How is the weather today in Hyderabad?",
        "expected": "Should politely refuse as this is strictly a financial assistant."
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