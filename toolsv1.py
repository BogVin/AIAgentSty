import os
import shutil
import json
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def clone_repo_tool(repo_url: str, local_path: str) -> str:
    """
    Clones a Git repository from a given URL to a specified local directory.
    """
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    try:
        os.system(f"git clone {repo_url} {local_path}")
        return f"Successfully cloned repository from {repo_url} to {local_path}."
    except Exception as e:
        return f"Failed to clone repository: {e}"


@tool
def list_files_tool(directory_path: str) -> str:
    """
    Lists all files in a given directory and its subdirectories.
    Returns a JSON string of file paths.
    """
    if not os.path.exists(directory_path):
        return json.dumps([])

    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return json.dumps(file_list)


@tool
def read_file_tool(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file {file_path}: {e}"


def clean_llm_output(raw: str) -> str:
    """
    Cleans LLM output by removing code fences and explanations.
    """
    cleaned = re.sub(r"^```(?:\w+)?", "", raw.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = re.sub(r"(?i)^here.*?\n+", "", cleaned.strip())
    return cleaned.strip()


@tool
def convert_code_tool(file_path: str, target_language: str, target_directory: str) -> str:
    """
    Converts code from a file into a specified target language using an LLM.
    The new file is saved in the target_directory.
    """
    extension_map = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "java": "java",
        "c++": "cpp",
        "c": "c",
    }

    llm_converter = init_chat_model(
        model="gemini-2.0-flash",
        temperature=0.3,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    )

    try:
        file_content = read_file_tool.invoke({"file_path": file_path})
        if "Failed to read file" in file_content:
            return file_content

        lang_key = target_language.lower().strip()
        if lang_key not in extension_map:
            return f"Unsupported target language: {target_language}"

        new_extension = extension_map[lang_key]

        prompt = (
            f"Convert the following code to {target_language}. "
            "Only return the converted code. No explanations. No markdown.\n\n"
            f"{file_content}"
        )

        response = llm_converter.invoke([HumanMessage(content=prompt)])
        code = clean_llm_output(response.content)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_path = os.path.join(target_directory, f"{base_name}.{new_extension}")
        os.makedirs(target_directory, exist_ok=True)

        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(code)

        return f"Successfully converted and saved code to {new_file_path}"

    except Exception as e:
        return f"Failed to convert code: {e}"
