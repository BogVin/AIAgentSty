# tools.py
import os
import re
import shutil
import subprocess
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ConversionResult(BaseModel):
    success: bool
    message: str
    python_files: Optional[list[str]] = Field(default=None)

CLONE_DIR = "/Users/bvintoni/work/aitest/test"

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def clean_llm_output(raw: str) -> str:
    """
    Cleans LLM output to remove code fences and commentary.
    Keeps only raw Python code.
    """
    # Remove ```python or ``` at start/end
    cleaned = re.sub(r"^```python\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    
    # Remove common explanations at the start
    explanation_patterns = [
        r"(?i)^Here's the converted Python code:\s*",
        r"(?i)^The equivalent Python code is:\s*",
        r"(?i)^Sure, here's the Python version:\s*"
    ]
    for pattern in explanation_patterns:
        cleaned = re.sub(pattern, "", cleaned.strip())

    return cleaned.strip()

@tool
def clone_and_convert_tool(github_url: str) -> ConversionResult:
    """
    Clones a Git repo and converts JavaScript files to Python using an LLM.
    """

    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)

    try:
        subprocess.run(["git", "clone", github_url, CLONE_DIR], check=True)
    except subprocess.CalledProcessError as e:
        return ConversionResult(success=False, message=f"Git clone failed: {e}")

    python_files = []

    for root, _, files in os.walk(CLONE_DIR):
        for f in files:
            if f.endswith(".js"):
                js_path = os.path.join(root, f)
                py_path = js_path[:-3] + ".py"

                try:
                    with open(js_path, "r", encoding="utf-8") as src:
                        js_code = src.read()

                    prompt = (
                        "Convert the following JavaScript code to clean, runnable Python code.\n"
                        "Do not include explanations or markdown. Just give the raw Python code.\n\n"
                        f"{js_code}"
                    )

                    response = model.invoke(prompt)

                    clean_python = clean_llm_output(response.content)

                    with open(py_path, "w", encoding="utf-8") as dst:
                        dst.write(f"# Converted from {f}\n")
                        dst.write(clean_python)

                    python_files.append(py_path)

                except Exception as e:
                    return ConversionResult(success=False, message=f"Conversion failed for {f}: {e}")

    return ConversionResult(
        success=True,
        message=f"Converted {len(python_files)} JS files to clean Python.",
        python_files=python_files
    )
