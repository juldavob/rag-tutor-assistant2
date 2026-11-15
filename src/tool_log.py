"""Registro de uso de tools: una línea JSON por pregunta en logs/uso_tools.jsonl."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import AGENT_CONFIG


class ToolLog:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or AGENT_CONFIG.log_path)

    def write(self, record: dict) -> None:
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # el registro nunca debe tumbar la respuesta
