from dataclasses import dataclass
from typing import List

@dataclass
class CliCommand:
    """Command for the CLI"""
    command: str
    description: str
    args: List[str]
    options: List[str]
    flags: List[str]
    examples: List[str]
    output: str
    error: str
    success: str
    failure: str

