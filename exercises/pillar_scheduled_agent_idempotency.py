#!/usr/bin/env python3
"""
Companion to: next/scheduled_agents.html
Minimal in-memory idempotency demo: duplicate "webhook" payloads do not double-apply.
"""

from __future__ import annotations

import hashlib
from typing import Any


def idempotency_key(payload: dict[str, Any]) -> str:
    """Stable key from business fields — NOT a random UUID per HTTP retry."""
    tenant = str(payload.get("tenant_id", ""))
    ev = str(payload.get("external_event_id", ""))
    raw = f"{tenant}:{ev}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> None:
    processed: set[str] = set()
    side_effects: list[str] = []

    def handle_event(payload: dict[str, Any]) -> str:
        key = idempotency_key(payload)
        if key in processed:
            return "duplicate_ignored"
        processed.add(key)
        side_effects.append(f"send_alert({key})")
        return "ok"

    p = {"tenant_id": "t1", "external_event_id": "evt-99", "data": {"x": 1}}
    assert handle_event(p) == "ok"
    assert handle_event(p) == "duplicate_ignored"
    assert handle_event(dict(p)) == "duplicate_ignored"
    assert len(side_effects) == 1
    print("OK: one side effect for three deliveries of the same logical event.")


if __name__ == "__main__":
    main()
