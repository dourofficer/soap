import os
import json
import re
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
from typing import Any, Dict, List, Optional, Tuple


SEEN = False
def print_once(text):
    global SEEN
    if not SEEN: 
        print(text)
        SEEN = True
        
def mdprint(text):
    console = Console()
    md = Markdown(text)
    console.print(md)

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _extract_metadata(data):
    """
    Extracts metadata for the labels field.
    Includes specific requested fields and any other metadata from the dataset,
    while excluding heavy fields like history and system_prompt.
    """
    metadata = {
        "question":           data.get("question"),
        "ground_truth":       data.get("ground_truth"),
        "is_corrected":       data.get("is_corrected"),
        "mistake_agent":      data.get("mistake_agent"),
        "mistake_step":       data.get("mistake_step"),
        "mistake_reason":     data.get("mistake_reason"),
        "mistake_type":       data.get("mistake_type"),
        "question_id":        data.get("question_ID"), 
        "system_description": data.get("system_prompt"),
        "subset":             data.get("subset"), # this is injected, not presented in original file.
        "filename":           data.get("filename"), # this as well is injected.
        "include_gt":         data.get("include_gt"), # this is injected during running inference.
    }
            
    return metadata
