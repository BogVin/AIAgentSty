import os
import shutil
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model

from toolsv1 import (
    clone_repo_tool,
    list_files_tool,
    read_files_tool,
    write_files_tool
)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    repo_url: str
    repo_path: str
    converted_path: str

all_tools = [clone_repo_tool, list_files_tool, read_files_tool, write_files_tool]

llm = init_chat_model(
    model="gemini-2.0-pro",  # or "gemini-2.0-flash"
    temperature=0,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
).bind_tools(all_tools)

def call_llm(state: AgentState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools=all_tools)

def decide_next_step(state: AgentState):
    last_msg = state['messages'][-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"

    return END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge("tool_node", "llm")

graph_builder.add_conditional_edges(
    "llm",
    decide_next_step,
    {
        "tool_node": "tool_node",
        END: END
    }
)

graph_builder.set_entry_point("llm")
graph = graph_builder.compile()

if __name__ == "__main__":
    TEST_DIR = "test"
    REPO_PATH = os.path.join(TEST_DIR, "cloned")
    CONVERTED_PATH = os.path.join(TEST_DIR, "converted")

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    os.makedirs(CONVERTED_PATH, exist_ok=True)

    initial_prompt = (
        f"Clone the repository from 'https://github.com/slashinfty/yt-frame-timer' "
        f"into '{REPO_PATH}', then read all files and convert the entire project "
        f"from JavaScript to Python as in expert JS to Python migration."
        "You must convert JS code to equivalent Python code using modern patterns, preserving ALL functionality" 
        f"Save the converted files to '{CONVERTED_PATH}'.\n\n"
        f"Please use the tools to clone, list files, read them, and write the converted output."
    )

    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)],
        "repo_url": "https://github.com/slashinfty/yt-frame-timer",
        "repo_path": REPO_PATH,
        "converted_path": CONVERTED_PATH
    }

    final_state = graph.invoke(initial_state)

    print("\nFinal state messages:")
    for msg in final_state["messages"]:
        print(f"- {msg}")
