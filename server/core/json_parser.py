"""Robust JSON parsing with fallback strategies for LLM output.

Robust JSON parsing with fallback strategies.
Handles thinking tokens, markdown fences, malformed JSON, and
provides a multi-attempt retry strategy with progressive constraints.
"""

import json
import logging
import re

log = logging.getLogger("prompt808.json_parser")


def parse_llm_json(raw_response, valid_keys=None):
    """Parse LLM response into a structured dict.

    Uses layered parsing:
    1. Strip thinking tokens (<think>...</think>)
    2. Strip markdown fences
    3. Try JSON parse (object or array)
    4. Regex fallback for malformed JSON

    If valid_keys is provided, filters out entries with unrecognized keys.
    Returns parsed result (dict or list) or empty dict on complete failure.
    """
    text = raw_response.strip()

    # Strip thinking tokens (Qwen3 Thinking models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        text = text.rsplit("```", 1)[0].strip()

    # Try 1: Extract JSON object {...}
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            parsed = json.loads(text[obj_start:obj_end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try 2: Extract JSON array [...]
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        try:
            parsed = json.loads(text[arr_start:arr_end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try 3: Regex fallback — extract type/desc pairs from malformed output
    desc_map = {}
    pairs = re.findall(r'"type"\s*:\s*"([^"]+)"[^}]*?"desc"\s*:\s*"([^"]+)"', text)
    if pairs:
        desc_map = {t: d for t, d in pairs}

    # Validate against known keys if provided
    if valid_keys and desc_map:
        desc_map = {k: v for k, v in desc_map.items() if k in valid_keys}

    return desc_map


def parse_element_extraction(raw_response):
    """Parse QwenVL element extraction response.

    Expects a JSON object with structure like:
    {
        "subject_type": "landscape",
        "elements": [
            {"category": "environment", "desc": "...", "tags": [...], "attributes": {...}},
            ...
        ]
    }

    Falls back through multiple parsing strategies.
    Returns (subject_type, elements_list) or (None, []) on failure.
    """
    parsed = parse_llm_json(raw_response)

    if isinstance(parsed, dict):
        subject_type = parsed.get("subject_type")
        elements = parsed.get("elements", [])
        if isinstance(elements, list) and elements:
            return subject_type, elements
        # Maybe the dict itself is a flat element list wrapper
        if "category" in parsed and "desc" in parsed:
            return parsed.get("subject_type"), [parsed]

    if isinstance(parsed, list):
        # Array of elements without wrapper
        return None, parsed

    log.warning("Failed to parse element extraction response")
    return None, []


def build_retry_prompt(base_prompt, attempt):
    """Build progressively stricter prompt for retry attempts.

    attempt 0: original prompt
    attempt 1: add strict JSON instruction
    attempt 2: add critical JSON-only instruction
    """
    if attempt == 0:
        return base_prompt
    elif attempt == 1:
        return base_prompt + (
            "\n\nIMPORTANT: Output ONLY valid JSON. "
            "No explanation, no markdown, no extra text."
        )
    else:
        return base_prompt + (
            "\n\nCRITICAL: You MUST output ONLY raw JSON starting with { or [ "
            "and ending with } or ]. No markdown fences, no commentary. Just the JSON."
        )
