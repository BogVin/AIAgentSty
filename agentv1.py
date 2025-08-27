import os
import shutil
import json
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model

from toolsv1 import (
    clone_repo_tool,
    list_files_tool,
    convert_code_tool
)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    file_queue: List[str]
    target_language: str
    repo_url: str
    repo_path: str
    converted_path: str

all_tools = [clone_repo_tool, list_files_tool, convert_code_tool]

llm = init_chat_model(
    model="gemini-2.0-flash",
    temperature=0,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
).bind_tools(all_tools)

def call_llm(state: AgentState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools=all_tools)

def process_file_list(state: AgentState):
    last_msg = state['messages'][-1]
    if isinstance(last_msg, ToolMessage):
        try:
            file_list = json.loads(last_msg.content)
            js_files = [f for f in file_list if f.endswith(".js")]
            return {
                "file_queue": js_files
            }
        except Exception as e:
            return {
                "file_queue": [],
                "messages": [HumanMessage(content=f"Failed to parse file list: {e}")]
            }
    return {}

def next_conversion_task(state: AgentState):
    if not state["file_queue"]:
        return {}

    next_file = state["file_queue"][0]
    remaining = state["file_queue"][1:]

    instruction = HumanMessage(content=json.dumps({
        "tool": "convert_code_tool",
        "tool_input": {
            "file_path": next_file,
            "target_language": state["target_language"],
            "target_directory": state["converted_path"]
        }
    }))

    return {
        "messages": [instruction],
        "file_queue": remaining
    }

def decide_next_step(state: AgentState):
    last_msg = state['messages'][-1]

    if isinstance(last_msg, ToolMessage):
        if len(state['messages']) >= 2 and isinstance(state['messages'][-2], AIMessage):
            ai_msg = state['messages'][-2]
            if ai_msg.tool_calls:
                tool_name = ai_msg.tool_calls[0].get("name")
                if tool_name == "clone_repo_tool":
                    return "tool_node"
                if tool_name == "list_files_tool":
                    return "process_file_list"
                if tool_name == "convert_code_tool" and state["file_queue"]:
                    return "next_conversion_task"

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"

    if state.get("file_queue"):
        return "next_conversion_task"

    return END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("process_file_list", process_file_list)
graph_builder.add_node("next_conversion_task", next_conversion_task)

graph_builder.add_edge("tool_node", "llm")
graph_builder.add_edge("process_file_list", "llm")
graph_builder.add_edge("next_conversion_task", "llm")

graph_builder.add_conditional_edges(
    "llm",
    decide_next_step,
    {
        "tool_node": "tool_node",
        "process_file_list": "process_file_list",
        "next_conversion_task": "next_conversion_task",
        END: END
    }
)

graph_builder.set_entry_point("llm")
graph = graph_builder.compile()

if __name__ == "__main__":
    TEST_DIR = "test"
    REPO_PATH = os.path.join(TEST_DIR, "cloned")
    CONVERTED_PATH = os.path.join(TEST_DIR, "converted")

    initial_state = {
        "messages": [
            HumanMessage(content="Clone the 'yt-frame-timer' repository from 'https://github.com/slashinfty/yt-frame-timer', then convert all '.js' files to Python. Save the cloned repo to 'test/cloned' and converted files to 'test/converted'.")
        ],
        "repo_url": "https://github.com/slashinfty/yt-frame-timer",
        "repo_path": REPO_PATH,
        "converted_path": CONVERTED_PATH,
        "target_language": "Python",
        "file_queue": []
    }

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    final_state = graph.invoke(initial_state)

    print("\nFinal state messages:")
    for msg in final_state["messages"]:
        print(f"- {msg}")
