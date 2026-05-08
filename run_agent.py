"""Run the lead lifecycle agent on one or more leads."""

import sys
from langchain_core.messages import HumanMessage, SystemMessage
from agent.agent import agent, AgentState
from agent.prompts import SYSTEM_PROMPT


def run_for_lead(lead_id: str):
    print(f"\n{'=' * 60}")
    print(f"  Processing Lead: {lead_id}")
    print(f"{'=' * 60}")

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Process lead {lead_id} and take the appropriate action."),
        ],
        "lead_id": lead_id,
    }

    for event in agent.stream(initial_state, stream_mode="values"):
        last = event["messages"][-1]
        last.pretty_print()

    print(f"\nFinished processing lead {lead_id}\n")


if __name__ == "__main__":
    lead_id = sys.argv[1] if len(sys.argv) > 1 else "L001"
    run_for_lead(lead_id)
