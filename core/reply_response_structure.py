"""Bounded, value-free diagnostics for Douyin reply response shapes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from utils.logger import setup_logger

logger = setup_logger("ReplyResponseStructure")
SCHEMA_VERSION = 1
MAX_DEPTH = 4
MAX_KEYS = 64
MAX_LIST_SAMPLES = 1
MAX_STRUCTURES = 8
MAX_FILE_BYTES = 32 * 1024

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "exception",
    "header",
    "html",
    "profile",
    "query",
    "session",
    "status_msg",
    "token",
    "traceback",
)
_ALLOWED_STATE_KEYS = frozenset({"status_code", "has_more", "cursor", "max_cursor"})
_ALLOWED_PRE_JSON_OUTCOMES = frozenset(
    {"empty", "non_json", "transport_error", "http_error"}
)


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    return "unsupported"


def _safe_key(value: object) -> str:
    key = str(value)
    lowered = key.lower()
    if not _FIELD_NAME_RE.fullmatch(key):
        return "<redacted>"
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    return key


def _state_value(key: str, value: object) -> Optional[object]:
    if key not in _ALLOWED_STATE_KEYS:
        return None
    if key == "has_more":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def build_reply_response_structure(payload: object) -> Dict[str, Any]:
    """Return a bounded response shape without retaining scalar strings."""

    fields: List[Dict[str, Any]] = []
    allowed_state: Dict[str, object] = {}
    field_count = 0
    recorded_key_count = 0

    def walk(value: object, path: str, depth: int, source_key: str = "") -> None:
        nonlocal field_count, recorded_key_count
        if field_count >= MAX_KEYS:
            return

        value_type = _json_type(value)
        field: Dict[str, Any] = {"path": path, "type": value_type}
        if isinstance(value, dict):
            keys: List[str] = []
            for raw_key in value:
                if recorded_key_count >= MAX_KEYS:
                    break
                safe_key = _safe_key(raw_key)
                if safe_key not in keys:
                    keys.append(safe_key)
                    recorded_key_count += 1
            field["keys"] = keys
        elif isinstance(value, list):
            field["length"] = len(value)
        fields.append(field)
        field_count += 1

        state = _state_value(source_key, value)
        if state is not None:
            allowed_state[path] = state

        if depth >= MAX_DEPTH or field_count >= MAX_KEYS:
            return
        if isinstance(value, dict):
            for raw_key, child in value.items():
                if field_count >= MAX_KEYS:
                    break
                safe_key = _safe_key(raw_key)
                walk(child, f"{path}.{safe_key}", depth + 1, str(raw_key))
        elif isinstance(value, list):
            for index, child in enumerate(value[:MAX_LIST_SAMPLES]):
                walk(child, f"{path}[{index}]", depth + 1)

    walk(payload, "$", 0)
    shape = {
        "root_type": _json_type(payload),
        "fields": fields,
    }
    canonical = json.dumps(shape, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION,
        **shape,
        "allowed_state": dict(sorted(allowed_state.items())),
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


class ReplyResponseStructureRecorder:
    """Keep a bounded set of reply response shapes in one atomic JSON file."""

    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self._lock = asyncio.Lock()
        self._capture_count = 0
        self._dropped_structure_count = 0
        self._structures: List[Dict[str, Any]] = []

    async def capture(self, payload: object) -> bool:
        try:
            structure = build_reply_response_structure(payload)
            return await self._capture_structure(structure)
        except Exception:  # noqa: BLE001 - diagnostics must never break collection
            logger.error("Reply response structure capture failed")
            return False

    async def capture_outcome(self, outcome: str) -> bool:
        safe_outcome = (
            outcome if outcome in _ALLOWED_PRE_JSON_OUTCOMES else "unknown"
        )
        canonical = json.dumps({"root_type": safe_outcome}, separators=(",", ":"))
        structure = {
            "schema_version": SCHEMA_VERSION,
            "root_type": safe_outcome,
            "fields": [],
            "allowed_state": {},
            "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        try:
            return await self._capture_structure(structure)
        except Exception:  # noqa: BLE001 - diagnostics must never break collection
            logger.error("Reply response structure capture failed")
            return False

    async def _capture_structure(self, structure: Dict[str, Any]) -> bool:
        async with self._lock:
            self._capture_count += 1
            fingerprint = structure["fingerprint"]
            existing = next(
                (
                    item
                    for item in self._structures
                    if item.get("fingerprint") == fingerprint
                ),
                None,
            )
            if existing is not None:
                existing["occurrence_count"] += 1
            elif len(self._structures) < MAX_STRUCTURES:
                sample = dict(structure)
                sample.pop("schema_version", None)
                sample["occurrence_count"] = 1
                self._structures.append(sample)
            else:
                self._dropped_structure_count += 1
            await self._write()
        return True

    def _artifact(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "capture_count": self._capture_count,
            "dropped_structure_count": self._dropped_structure_count,
            "structures": self._structures,
        }

    @staticmethod
    def _serialize(artifact: Dict[str, Any]) -> bytes:
        return (
            json.dumps(
                artifact,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    async def _write(self) -> None:
        artifact = self._artifact()
        body = self._serialize(artifact)
        while len(body) > MAX_FILE_BYTES and self._structures:
            self._structures.pop()
            self._dropped_structure_count += 1
            artifact = self._artifact()
            body = self._serialize(artifact)
        if len(body) > MAX_FILE_BYTES:
            raise ValueError("reply response structure artifact exceeds fixed limit")

        await asyncio.to_thread(self.destination.parent.mkdir, parents=True, exist_ok=True)
        temporary = self.destination.with_name(f".{self.destination.name}.tmp")
        try:
            async with aiofiles.open(temporary, "wb") as handle:
                await handle.write(body)
                await handle.flush()
            await asyncio.to_thread(os.replace, temporary, self.destination)
        finally:
            if await asyncio.to_thread(temporary.exists):
                await asyncio.to_thread(temporary.unlink)
