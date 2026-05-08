"""
Simulate a call outcome being logged by sending a message to SQS.
This represents what would happen after a sales rep finishes a call.

Usage:
    python aws/publish_event.py L001
    python aws/publish_event.py L005 converted
"""

import os
import sys
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ["AWS_REGION"]
QUEUE_URL = os.environ["SQS_QUEUE_URL"]

VALID_LEADS = ["L001", "L002", "L003", "L004", "L005",
               "L006", "L007", "L008", "L009", "L010"]


def publish(lead_id: str, outcome: str = "voicemail"):
    sqs = boto3.client("sqs", region_name=REGION)

    message = {
        "lead_id": lead_id,
        "event": "call_completed",
        "outcome": outcome,
    }

    response = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message),
    )

    print(f"Message sent for lead {lead_id} (outcome: {outcome})")
    print(f"MessageId: {response['MessageId']}")


if __name__ == "__main__":
    lead_id = sys.argv[1] if len(sys.argv) > 1 else "L001"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "voicemail"

    if lead_id not in VALID_LEADS:
        print(f"Unknown lead: {lead_id}. Valid leads: {VALID_LEADS}")
        sys.exit(1)

    publish(lead_id, outcome)
