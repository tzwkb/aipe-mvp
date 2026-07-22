"""Parse and normalize structured LLM translation output."""

from __future__ import annotations

import json
import re

from app.schemas.web_search import WebSearchResult


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.\)）。、:：]\s*(.+?)\s*$")


def parse_single_llm_output(raw: str) -> tuple[str, str | None]:
    """Extract a single translation and optional reason from an LLM response."""
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "translation" in data:
            translation = str(data["translation"]).strip()
            reason = str(data.get("reason", "") or "").strip() or None
            return translation, reason
    except (json.JSONDecodeError, ValueError):
        pass
    result = _extract_translation_reason(text)
    if result is not None:
        return result
    return text, None


def parse_numbered_output_positional(
    raw: str, expected: int
) -> list[tuple[str, str | None]] | None:
    """Parse exactly ``expected`` numbered lines while ignoring number values."""
    if not raw or expected <= 0:
        return None
    items: list[tuple[str, str | None]] = []
    for line in raw.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if not match:
            continue
        text = match.group(2).strip()
        if text:
            items.append(_parse_line_text(text))
    if len(items) != expected:
        return None
    return items


def parse_numbered_output(
    raw: str, expected: int
) -> list[tuple[str, str | None]] | None:
    """Parse numbered lines whose indexes must be exactly ``1..expected``."""
    if not raw or expected <= 0:
        return None
    found: dict[int, tuple[str, str | None]] = {}
    for line in raw.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        text = match.group(2).strip()
        if text:
            found.setdefault(index, _parse_line_text(text))
    if len(found) != expected or set(found) != set(range(1, expected + 1)):
        return None
    return [found[index] for index in range(1, expected + 1)]


def extract_image_analysis(web_refs: list[WebSearchResult] | None) -> str | None:
    """Return the first non-empty image analysis from web references."""
    if not web_refs:
        return None
    return next((ref.image_analysis for ref in web_refs if ref.image_analysis), None)


def _parse_line_text(text: str) -> tuple[str, str | None]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "translation" in data:
            translation = str(data["translation"]).strip()
            reason = str(data.get("reason", "") or "").strip() or None
            return translation, reason
    except (json.JSONDecodeError, ValueError):
        pass
    result = _extract_translation_reason(text)
    if result is not None:
        return result
    return text, None


def _extract_translation_reason(text: str) -> tuple[str, str | None] | None:
    """Extract malformed JSON fields containing unescaped quotes as a fallback."""
    translation_match = re.search(r'"translation"\s*:\s*"([^"]*)"', text)
    if not translation_match:
        return None
    reason_match = re.search(r'"reason"\s*:\s*"(.*)"[\s\n]*\}', text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else None
    return translation_match.group(1).strip(), reason or None


__all__ = [
    "extract_image_analysis",
    "parse_numbered_output",
    "parse_numbered_output_positional",
    "parse_single_llm_output",
]
