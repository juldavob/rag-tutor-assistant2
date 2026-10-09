"""Hace importables los módulos de src/ desde los scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # evita errores de tildes en terminales Windows
    sys.stdout.reconfigure(encoding="utf-8")
