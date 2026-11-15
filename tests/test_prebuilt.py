"""Índice precalculado: la nube debe poder cargarlo SIN documentos en knowledge_base/."""
import pytest

from config import KNOWLEDGE_BASE_DIR
from embeddings import HashEmbedder
from retriever import Retriever
from vector_store import build_prebuilt_store, has_prebuilt, load_store_auto


def test_prebuilt_roundtrip_without_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_MODE", "auto")
    embedder, pre = HashEmbedder(), tmp_path / "pre"
    built = build_prebuilt_store(embedder, 700, 100, path=pre)
    assert has_prebuilt(pre) and built.count() > 0
    empty_kb = tmp_path / "kb_vacia"
    empty_kb.mkdir()
    loaded = load_store_auto(embedder, 700, 100, kb_dir=empty_kb, persist_dir=tmp_path / "local", prebuilt_dir=pre)
    assert loaded.count() == built.count()
    assert Retriever(loaded, embedder, 3).retrieve("error tipo I")[0].source.endswith(".md")


def test_with_documents_uses_normal_local_index(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_MODE", "auto")
    embedder = HashEmbedder()
    build_prebuilt_store(embedder, 700, 100, path=tmp_path / "pre")
    store = load_store_auto(embedder, 300, 30, kb_dir=KNOWLEDGE_BASE_DIR, persist_dir=tmp_path / "local", prebuilt_dir=tmp_path / "pre")
    assert store.count() != 0 and (tmp_path / "local").exists()   # respetó chunk_size=300 (no usó el precalculado)


def test_forced_prebuilt_and_embedder_mismatch(tmp_path, monkeypatch):
    embedder = HashEmbedder()
    build_prebuilt_store(embedder, 700, 100, path=tmp_path / "pre")
    other = HashEmbedder(); other.name = "minilm"
    monkeypatch.setenv("INDEX_MODE", "prebuilt")
    with pytest.raises(RuntimeError, match="embeddings"):
        load_store_auto(other, 700, 100, prebuilt_dir=tmp_path / "pre")
    with pytest.raises(RuntimeError, match="index_prebuilt"):
        load_store_auto(embedder, 700, 100, prebuilt_dir=tmp_path / "no_existe")
