"""Configura backends de prueba (sin red ni descargas) antes de importar el código."""
import os
import sys
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["LLM_BACKEND"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
