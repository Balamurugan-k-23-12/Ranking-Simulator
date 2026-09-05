"""Student embedder: mock hash offline, or MiniLM when schemas say so.

Vendored into the student bundle. When ``provider`` is ``mock_hash``, vectors match
the instructor mock embedder for the same model_name, dim, and synonym_group_weight.
When ``provider`` is ``sentence_transformers``, encode uses the same pinned MiniLM
model as the instructor catalogue so query vectors sit in the published e_i space.

Standard library + numpy for mock_hash. sentence_transformers is imported only when
that provider is selected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_DIGEST_SIZE = 32
_EPS = 1e-12
_PINNED_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PINNED_ST_DIM = 384

# Synonym groups of the synthetic catalogue. Two words in one group share most of their
# vector, so a query that says "earbuds" matches an item that says "headphones".
CATEGORY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("headphones", ("headphones", "earphones", "earbuds", "headset", "cans")),
    ("keyboard", ("keyboard", "keypad", "typeboard", "keyset")),
    ("mouse", ("mouse", "trackball", "pointer", "clicker")),
    ("monitor", ("monitor", "display", "screen", "panel")),
    ("laptop", ("laptop", "notebook", "ultrabook", "portablepc")),
    ("speaker", ("speaker", "soundbar", "loudspeaker", "audiounit")),
    ("camera", ("camera", "webcam", "camcorder", "imager")),
    ("charger", ("charger", "adapter", "powerbrick", "psu")),
    ("backpack", ("backpack", "rucksack", "daypack", "knapsack")),
    ("kettle", ("kettle", "boiler", "teapot", "waterboiler")),
    ("blender", ("blender", "mixer", "liquidiser", "smoothiejar")),
    ("lamp", ("lamp", "light", "luminaire", "glowlamp")),
)

ATTRIBUTE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cheap", ("cheap", "budget", "affordable", "inexpensive", "lowcost")),
    ("sturdy", ("sturdy", "durable", "rugged", "tough", "robust")),
    ("compact", ("compact", "small", "portable", "pocketable", "tiny")),
    ("fast", ("fast", "quick", "rapid", "speedy", "swift")),
    ("quiet", ("quiet", "silent", "noiseless", "hushed", "muted")),
    ("bright", ("bright", "vivid", "luminous", "brilliant", "radiant")),
    ("light", ("light", "lightweight", "featherweight", "airy", "ultralight")),
    ("warm", ("warm", "cosy", "snug", "toasty", "heated")),
    ("sealed", ("sealed", "waterproof", "watertight", "splashproof", "insulated")),
    ("wireless", ("wireless", "cordless", "untethered", "cablefree", "unwired")),
    ("smart", ("smart", "clever", "intelligent", "connected", "programmable")),
    ("premium", ("premium", "deluxe", "upscale", "luxury", "highend")),
    ("simple", ("simple", "plain", "minimal", "basic", "unadorned")),
    ("roomy", ("roomy", "spacious", "capacious", "wide", "ample")),
)

_WORD_TO_GROUP: dict[str, str] = {
    word: name
    for name, words in (CATEGORY_GROUPS + ATTRIBUTE_GROUPS)
    for word in words
}

# Process-level SentenceTransformer handles, keyed by model name.
_ST_MODELS: dict[str, object] = {}


@dataclass(frozen=True)
class EmbedderConfig:
    """The knobs that decide the vectors. Read them from schemas.json."""
    provider: str = "mock_hash"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 128
    synonym_group_weight: float = 0.6
    normalize: bool = True

    @classmethod
    def from_schemas(cls, schemas: dict | str | Path) -> "EmbedderConfig":
        if isinstance(schemas, (str, Path)):
            body = json.loads(Path(schemas).read_text(encoding="ascii"))
        else:
            body = schemas
        section = body["embedder"] if "embedder" in body else body
        return cls(
            provider=str(section.get("provider", "mock_hash")),
            model_name=str(section["model_name"]),
            dim=int(section["dim"]),
            synonym_group_weight=float(section["synonym_group_weight"]),
            normalize=bool(section["normalize"]),
        )


def tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def group_of(word: str) -> str | None:
    return _WORD_TO_GROUP.get(word.lower())


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, _EPS)


def make_embedder(config: EmbedderConfig | dict | None = None):
    """Build the embedder named in schemas (mock_hash or sentence_transformers)."""
    section = _as_config(config)
    if section.provider == "mock_hash":
        return MockHashEmbedder(section)
    if section.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(section)
    raise ValueError(
        f"unknown embedding provider {section.provider!r}; "
        f"have mock_hash, sentence_transformers")


class MockHashEmbedder:
    """Deterministic offline embedder. No download, and no RNG at call time."""

    def __init__(self, config: EmbedderConfig | dict | None = None) -> None:
        section = _as_config(config)
        weight = float(section.synonym_group_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"synonym_group_weight must be in [0, 1], got {weight}")
        self.config = section
        self.name = f"mock_hash:{section.model_name}"
        self.dim = int(section.dim)
        self._group_weight = weight
        self._normalize = bool(section.normalize)
        self._cache: dict[str, np.ndarray] = {}

    def word_vector(self, word: str) -> np.ndarray:
        key = word.lower()
        vector = self._cache.get(key)
        if vector is None:
            vector = self._build_word_vector(key)
            self._cache[key] = vector
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._document_vector(text) for text in texts]
        matrix = np.vstack(rows) if rows else np.zeros((0, self.dim))
        return unit_rows(matrix) if self._normalize else matrix

    def _build_word_vector(self, word: str) -> np.ndarray:
        token_part = _hash_vector(self.config.model_name, f"token:{word}", self.dim)
        group = group_of(word)
        if group is None:
            return token_part
        group_part = _hash_vector(self.config.model_name, f"group:{group}", self.dim)
        weight = self._group_weight
        return np.sqrt(weight) * group_part + np.sqrt(1.0 - weight) * token_part

    def _document_vector(self, text: str) -> np.ndarray:
        words = tokenize(text)
        if not words:
            return np.zeros(self.dim)
        return np.mean([self.word_vector(word) for word in words], axis=0)


class SentenceTransformerEmbedder:
    """Pinned open-source CPU MiniLM. Matches the instructor catalogue space."""

    def __init__(self, config: EmbedderConfig | dict | None = None) -> None:
        section = _as_config(config)
        self.config = section
        self.name = f"sentence_transformers:{section.model_name}"
        self._normalize = bool(section.normalize)
        self._model = None
        if section.model_name == _PINNED_ST_MODEL:
            self.dim = _PINNED_ST_DIM
        else:
            model = _sentence_transformer_model(section.model_name)
            self._model = model
            self.dim = int(_model_dim(model))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = list(texts)
        if not rows:
            return np.zeros((0, self.dim))
        model = self._ensure_model()
        matrix = np.asarray(
            model.encode(rows, convert_to_numpy=True),
            dtype=np.float64,
        ).reshape(len(rows), -1)
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"model {self.config.model_name!r} width {matrix.shape[1]} != "
                f"embedder.dim {self.dim}")
        return unit_rows(matrix) if self._normalize else matrix

    def _ensure_model(self):
        if self._model is None:
            self._model = _sentence_transformer_model(self.config.model_name)
            live_dim = int(_model_dim(self._model))
            if live_dim != self.dim:
                raise ValueError(
                    f"model {self.config.model_name!r} width {live_dim} != "
                    f"embedder.dim {self.dim}")
        return self._model


def _as_config(config: EmbedderConfig | dict | None) -> EmbedderConfig:
    if config is None:
        return EmbedderConfig()
    if isinstance(config, dict):
        return EmbedderConfig(
            provider=str(config.get("provider", EmbedderConfig.provider)),
            model_name=str(config.get("model_name", EmbedderConfig.model_name)),
            dim=int(config.get("dim", EmbedderConfig.dim)),
            synonym_group_weight=float(
                config.get("synonym_group_weight",
                           EmbedderConfig.synonym_group_weight)),
            normalize=bool(config.get("normalize", EmbedderConfig.normalize)),
        )
    return config


def _sentence_transformer_model(model_name: str):
    handle = _ST_MODELS.get(model_name)
    if handle is None:
        from sentence_transformers import SentenceTransformer
        handle = SentenceTransformer(model_name)
        _ST_MODELS[model_name] = handle
    return handle


def _model_dim(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def _hash_vector(model_name: str, name: str, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{model_name}\x1f{name}".encode("utf-8"),
                             digest_size=_DIGEST_SIZE).digest()
    return np.random.default_rng(int.from_bytes(digest, "big")).normal(size=dim)
