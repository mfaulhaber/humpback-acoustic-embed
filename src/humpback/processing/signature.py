import hashlib
import json
from typing import Any, Optional


def compute_encoding_signature(
    model_version: str,
    window_size_seconds: float,
    target_sample_rate: int,
    feature_config: Optional[dict[str, Any]] = None,
) -> str:
    """Compute a deterministic SHA-256 signature for an encoding configuration."""
    payload = {
        "model_version": model_version,
        "window_size_seconds": window_size_seconds,
        "target_sample_rate": target_sample_rate,
        "feature_config": feature_config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
