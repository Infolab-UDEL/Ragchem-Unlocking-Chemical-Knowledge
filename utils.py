import re
import json
import os
from typing import Literal

def cleanDocument(doc):
    # This regex pattern captures square brackets with numbers inside, allowing optional spaces and commas.
    pattern = r"\[\s*\d+\s*(,\s*\d+\s*)*\]"
    cleaned_text = re.sub(pattern, "", doc)
    cleaned_text = cleaned_text.replace("<hi>", "").replace("</hi>", "")
    return cleaned_text.strip()


def models_available_llm(key:Literal["olmo","falcon","qwen","phi","llama","deepSmall","deepLarge","gpt" ,"gemma"])-> str:
    """
    Available are 
    """
    models_available = {
    "olmo": "olmo2:13b",
    "falcon": "falcon3:10b",
    "qwen": "qwen2.5:14b",
    "phi": "phi4",
    "llama": "llama3.1",
    "deepSmall" : "deepseek-r1:8b",
    "deepLarge" : "deepseek-r1:14b",
    "gpt" : "gpt-4o-mini",
    "gemma": "gemma3:12b"
    }
    assert key in models_available, f"Model '{key}' not found in dictionary of available LLMs"
    return models_available[key]


def append_to_jsonl(file_path, data):
    """Append a new entry to a JSONL file."""
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")

def append_to_jsonl_v2(file_path, name, data):
    """Append a new entry to a JSONL file."""
    full_path = os.path.join(file_path, name)
    with open(full_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")

#Now with the filter documents I want to get the metrics for the True False questions
def read_jsonl(file_path):
    """Read and return all records from a JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line.strip()))
    return records

def read_jsonl_V2(file_path , name):
    """Read and return all records from a JSONL file."""
    full_path = os.path.join(file_path, name)
    records = []
    with open(full_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line.strip()))
    return records