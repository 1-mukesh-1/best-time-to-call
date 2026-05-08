"""
SQS worker: polls the lead-call-events queue and triggers the LangGraph agent
for each incoming message.

Usage:
    python aws/worker.py
"""

import os
import json
import boto3
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

REGION = os.environ["AWS_REGION"]
QUEUE_URL = os.environ["SQS_QUEUE_URL"]

POLL_WAIT_SECONDS = 10  # long polling — reduces empty receives


def process_message(body: dict):
    lead_id = body["lead_id"]
    outcome = body.get("outcome", "unknown")

    print(f"\n{'=' * 60}")
    print(f"  Event received — Lead: {lead_id} | Outcome: {outcome}")
    print(f"{'=' * 60}")

    # Import here to avoid loading at startup before env vars are set
    from agent.agent import agent, AgentState
    from agent.prompts import SYSTEM_PROMPT

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"A call just completed for lead {lead_id} with outcome: {outcome}. "
                    f"Review their full history and decide the next action."
                )
            ),
        ],
        "lead_id": lead_id,
    }

    for event in agent.stream(initial_state, stream_mode="values"):
        last = event["messages"][-1]
        last.pretty_print()

    print(f"\nDone processing lead {lead_id}\n")


def run():
    sqs = boto3.client("sqs", region_name=REGION)
    print(f"Worker started. Polling queue: {QUEUE_URL}\n")

    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=POLL_WAIT_SECONDS,
        )

        messages = response.get("Messages", [])
        if not messages:
            print("No messages. Waiting...")
            continue

        for msg in messages:
            try:
                body = json.loads(msg["Body"])
                process_message(body)
                # Delete from queue only after successful processing
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"],
                )
            except Exception as e:
                print(f"Error processing message: {e}")
                # Message becomes visible again after VisibilityTimeout


if __name__ == "__main__":
    run()
