"""Four-pass humpback call parsing pipeline.

Phase 0 scaffold: dataclasses, parquet I/O helpers, and reserved
subpackage namespaces for Passes 1–4. See
``docs/specs/2026-04-11-call-parsing-pipeline-phase0-design.md`` for the
architecture contract.
"""

from humpback.call_parsing.types import (
    EVENT_SCHEMA,
    REGION_SCHEMA,
    TRACE_SCHEMA,
    TYPED_EVENT_SCHEMA,
    Event,
    Region,
    TypedEvent,
    WindowScore,
    new_uuid,
)

__all__ = [
    "Event",
    "EVENT_SCHEMA",
    "Region",
    "REGION_SCHEMA",
    "TRACE_SCHEMA",
    "TypedEvent",
    "TYPED_EVENT_SCHEMA",
    "WindowScore",
    "new_uuid",
]
