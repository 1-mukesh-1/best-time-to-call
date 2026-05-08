from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import (
    get_lead_info,
    get_call_logs,
    predict_best_time,
    schedule_call,
    escalate_to_human,
    disqualify_lead,
)
from agent.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    lead_id: str


TOOLS = [
    get_lead_info,
    get_call_logs,
    predict_best_time,
    schedule_call,
    escalate_to_human,
    disqualify_lead,
]

def reason(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


tool_node = ToolNode(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("reason", reason)
builder.add_node("tools", tool_node)
builder.add_edge(START, "reason")
builder.add_conditional_edges("reason", should_continue)
builder.add_edge("tools", "reason")

agent = builder.compile()
