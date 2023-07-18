"""IO utils"""
from pathlib import Path


def get_project_root() -> Path:
    """Returns project root path"""
    return Path(__file__).parents[2]
