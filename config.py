from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
# Set EMBEDDER=bge-m3 to use the local model instead of the OpenAI API.
EMBEDDER = os.getenv("EMBEDDER", "openai").lower()

OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_EMBED_DIMS = 1024  # 3072 = max quality; 1024 = excellent + smaller index
BGE_MODEL = "BAAI/bge-m3"

GRAMMAR_COLLECTION = "grammar_rules"
VOCAB_COLLECTION = "vocabulary"

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Embeds text using text-embedding-3-large via the OpenAI API.
    Reads OPENAI_API_KEY from the environment. Each runtime query makes one API call
    to embed the student message before retrieval."""

    def __init__(self, model: str = OPENAI_EMBED_MODEL,
                 dimensions: int = OPENAI_EMBED_DIMS, api_key: str | None = None):
        from openai import OpenAI
        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        resp = self._client.embeddings.create(
            model=self.model, input=list(input), dimensions=self.dimensions
        )
        return [d.embedding for d in resp.data]

    @staticmethod
    def name() -> str:
        return "openai_text_embedding_3_large"

    def get_config(self) -> dict:
        return {"model": self.model, "dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(
            model=config.get("model", OPENAI_EMBED_MODEL),
            dimensions=config.get("dimensions", OPENAI_EMBED_DIMS),
        )


class BGEM3EmbeddingFunction(EmbeddingFunction):
    """Embeds text locally using BAAI/bge-m3 via sentence-transformers.
    Zero per-query cost and fully offline; requires ~2 GB model download."""

    _model = None

    def __init__(self, model_name: str = BGE_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device

    def _get_model(self):
        if BGEM3EmbeddingFunction._model is None:
            from sentence_transformers import SentenceTransformer
            BGEM3EmbeddingFunction._model = SentenceTransformer(
                self.model_name, device=self.device
            )
        return BGEM3EmbeddingFunction._model

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._get_model().encode(
            list(input), normalize_embeddings=True, batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @staticmethod
    def name() -> str:
        return "bge_m3"

    def get_config(self) -> dict:
        return {"model_name": self.model_name, "device": self.device}

    @staticmethod
    def build_from_config(config: dict) -> "BGEM3EmbeddingFunction":
        return BGEM3EmbeddingFunction(
            model_name=config.get("model_name", BGE_MODEL),
            device=config.get("device"),
        )


def get_embedding_function() -> EmbeddingFunction:
    if EMBEDDER == "bge-m3":
        return BGEM3EmbeddingFunction()
    return OpenAIEmbeddingFunction()


def get_client(persist_dir: str = PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def get_collections(client: chromadb.ClientAPI | None = None):
    """Returns (grammar_collection, vocab_collection), creating them if needed.
    Ingest and query must use the same embedder — mixing embedders corrupts retrieval."""
    client = client or get_client()
    ef = get_embedding_function()
    grammar = client.get_or_create_collection(
        name=GRAMMAR_COLLECTION, embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    vocab = client.get_or_create_collection(
        name=VOCAB_COLLECTION, embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return grammar, vocab


def validate_metadata(meta: dict, *, kind: str) -> dict:
    """Validates required keys and CEFR level before inserting into ChromaDB.
    Coerces None values to empty strings (Chroma only accepts str/int/float/bool)."""
    required = {"language", "cefr_level", "topic", "type", "source"}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"metadata missing keys {missing}: {meta}")
    if meta["cefr_level"] not in CEFR_LEVELS:
        raise ValueError(f"bad cefr_level {meta['cefr_level']!r}; use {CEFR_LEVELS}")
    if meta["type"] != kind:
        raise ValueError(f"type {meta['type']!r} != expected {kind!r}")
    return {k: ("" if v is None else v) for k, v in meta.items()}
