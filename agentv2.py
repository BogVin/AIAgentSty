import os
import shutil
import operator
from typing import TypedDict, Sequence, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

from toolsv1 import clone_repo_tool, list_files_tool, read_files_tool, write_files_tool

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    repo_url: str
    repo_path: str
    converted_path: str

all_tools = [clone_repo_tool, list_files_tool, read_files_tool, write_files_tool]
tool_node = ToolNode(tools=all_tools)

llm = init_chat_model(
    model="google_genai:gemini-2.0-flash",
    temperature=0,
).bind_tools(all_tools)

def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def decide_next_step(state: AgentState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"
    return END

graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_edge("tool_node", "llm")
graph_builder.add_conditional_edges("llm", decide_next_step, {
    "tool_node": "tool_node",
    END: END
})

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
        f"You are a world-class code migration assistant.\n\n"
        f"Convert the JavaScript in-memory database library hosted at the following GitHub repository:\n"
        f" https://github.com/tducasse/js-db\n\n"
        f"Steps:\n"
        f"1. Clone the repository from the given URL into the local path: `{REPO_PATH}`.\n"
        f"2. List all JavaScript files recursively from the cloned repo.\n"
        f"3. Read the contents of each JavaScript file.\n"
        f"4. Analyze the contents and determine how to convert each file to Python.\n"
        f"5. Convert the code while maintaining functionality and preserving file structure.\n"
        f"6. Save converted files in the `{CONVERTED_PATH}` directory with the same relative structure.\n"
        f"7. Ensure functionality such as insert/find/update/remove is preserved.\n"
        f"8. Maintain collection registration, dot-notation support, and business logic.\n"
        f"9. Generate idiomatic, clean, and well-documented Python code.\n"
        f"10. If appropriate, create a small Flask API exposing similar functionality.\n"
        f"\nUse the available tools to accomplish these tasks. You can call tools multiple times as needed."
    )

    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)],
        "repo_url": "https://github.com/tducasse/js-db",
        "repo_path": REPO_PATH,
        "converted_path": CONVERTED_PATH,
    }

    final_state = graph.invoke(initial_state)

    print("\n Final Output Messages:")
    for msg in final_state["messages"]:
        print(f"- {getattr(msg, 'content', msg)}")
