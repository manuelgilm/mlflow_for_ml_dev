from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parents[3]