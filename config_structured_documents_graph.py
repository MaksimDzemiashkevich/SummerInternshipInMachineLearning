from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    input_path_document: Path = Path("API-600.pdf")