import os
import shutil
import json
import re
from dotenv import load_dotenv
from langchain_core.tools import tool

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
        return f"Cloned repository from {repo_url} to {local_path}."
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
def read_files_tool(file_paths: list) -> dict:
    """
    Reads content of multiple files.
    Returns a dictionary { file_path: content }.
    """
    result = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                result[path] = f.read()
        except Exception as e:
            result[path] = f"Error reading file: {e}"
    return result

@tool
def write_files_tool(files: dict, base_path: str) -> str:
    """
    Writes multiple files. 
    `files` is a dict where key = relative path, value = content.
    """
    for rel_path, content in files.items():
        abs_path = os.path.join(base_path, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Failed to write {rel_path}: {e}"
    return f"Successfully wrote {len(files)} files to {base_path}"
