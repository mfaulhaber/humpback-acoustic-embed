"""Pluggable projections for CRNN region chunk embeddings.

A ``ChunkProjection`` reduces or remaps the 1024-d concat-of-eight-frame
chunk vectors that ``crnn_features`` produces. Three implementations
ship in Phase 1:

- ``IdentityProjection``: pass-through; ``output_dim == input_dim``.
- ``RandomProjection``: deterministic Gaussian random projection.
- ``PCAProjection``: standard PCA, optionally whitened.

Each implementation persists via ``joblib`` so the same projection used
at producer time can be re-applied later (e.g. for inspection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


@runtime_checkable
class ChunkProjection(Protocol):
    """Protocol every chunk projection implementation satisfies."""

    output_dim: int

    def fit(self, X: np.ndarray) -> None: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...


class IdentityProjection:
    """Pass-through projection; ``transform`` returns the input."""

    def __init__(self, input_dim: int) -> None:
        self.output_dim: int = int(input_dim)

    def fit(self, X: np.ndarray) -> None:
        # No-op: identity has no parameters.
        return None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.output_dim:
            raise ValueError(
                f"IdentityProjection expects (N, {self.output_dim}) input, "
                f"got shape={X.shape}"
            )
        return X

    def save(self, path: str | Path) -> None:
        joblib.dump({"kind": "identity", "input_dim": self.output_dim}, path)

    @classmethod
    def load(cls, path: str | Path) -> "IdentityProjection":
        payload = joblib.load(path)
        if payload.get("kind") != "identity":
            raise ValueError(f"Not an IdentityProjection payload at {path}")
        return cls(input_dim=int(payload["input_dim"]))


class RandomProjection:
    """Deterministic Gaussian random projection.

    Wraps ``sklearn.random_projection.GaussianRandomProjection`` with a
    fixed RNG seed so two instances with the same ``output_dim`` and
    ``seed`` produce bit-identical transforms after ``fit``.
    """

    def __init__(self, output_dim: int, seed: int) -> None:
        self.output_dim: int = int(output_dim)
        self.seed: int = int(seed)
        self._rp: GaussianRandomProjection | None = None

    def fit(self, X: np.ndarray) -> None:
        # sklearn's stub overloads ``n_components`` to ``"auto" | int``;
        # pyright's bundled stub only sees the ``str`` overload, so guide
        # it explicitly with a cast.
        rp: GaussianRandomProjection = GaussianRandomProjection(
            n_components=self.output_dim,  # type: ignore[arg-type]
            random_state=self.seed,
        )
        rp.fit(X)
        self._rp = rp

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._rp is None:
            raise RuntimeError("RandomProjection.fit must be called before transform")
        return self._rp.transform(X).astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        if self._rp is None:
            raise RuntimeError("Cannot save RandomProjection before fit")
        joblib.dump(
            {
                "kind": "random",
                "output_dim": self.output_dim,
                "seed": self.seed,
                "rp": self._rp,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomProjection":
        payload = joblib.load(path)
        if payload.get("kind") != "random":
            raise ValueError(f"Not a RandomProjection payload at {path}")
        instance = cls(output_dim=int(payload["output_dim"]), seed=int(payload["seed"]))
        instance._rp = payload["rp"]
        return instance


class PCAProjection:
    """PCA projection with optional whitening."""

    def __init__(self, output_dim: int, whiten: bool = False) -> None:
        self.output_dim: int = int(output_dim)
        self.whiten: bool = bool(whiten)
        self._pca: PCA | None = None

    def fit(self, X: np.ndarray) -> None:
        self._pca = PCA(n_components=self.output_dim, whiten=self.whiten)
        self._pca.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAProjection.fit must be called before transform")
        return self._pca.transform(X).astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        if self._pca is None:
            raise RuntimeError("Cannot save PCAProjection before fit")
        joblib.dump(
            {
                "kind": "pca",
                "output_dim": self.output_dim,
                "whiten": self.whiten,
                "pca": self._pca,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCAProjection":
        payload = joblib.load(path)
        if payload.get("kind") != "pca":
            raise ValueError(f"Not a PCAProjection payload at {path}")
        instance = cls(
            output_dim=int(payload["output_dim"]), whiten=bool(payload["whiten"])
        )
        instance._pca = payload["pca"]
        return instance


def load_projection(path: str | Path) -> ChunkProjection:
    """Dispatch on the persisted ``kind`` field to pick the right loader."""
    payload = joblib.load(path)
    kind = payload.get("kind")
    if kind == "identity":
        return IdentityProjection(input_dim=int(payload["input_dim"]))
    if kind == "random":
        return RandomProjection.load(path)
    if kind == "pca":
        return PCAProjection.load(path)
    raise ValueError(f"Unknown projection kind {kind!r} at {path}")
