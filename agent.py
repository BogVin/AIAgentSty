from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from tools import clone_and_convert_tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    # Initialize the chat model with tools
    agent = create_react_agent(
        model=init_chat_model("google_genai:gemini-2.0-flash"),
        tools=[clone_and_convert_tool],
        prompt="You are a helpful assistant that clones a repo and converts JS to Python."
    )

    # Example user prompt
    user_message = HumanMessage(content="Clone the repo https://github.com/slashinfty/yt-frame-timer and convert JS to Python.")

    # Invoke agent
    result = agent.invoke({"messages": [user_message]})
    print(result)

if __name__ == "__main__":
    main()
