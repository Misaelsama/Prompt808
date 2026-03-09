"""Prompt808 Library Select Node — multi-library selection for generation.

Provides a Power Lora Loader-inspired interface with dynamic library slots.
Each slot has a toggle and a library dropdown. The node outputs a list of
enabled library names that the Prompt808 Generate node uses to merge
elements from multiple libraries.
"""

import logging

log = logging.getLogger("prompt808.library_select")


class _AnyType(str):
    """Matches any ComfyUI type for flexible input connections."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


_any_type = _AnyType("*")


class _FlexibleInputType(dict):
    """Dict subclass that accepts any key, enabling dynamic inputs.

    Known keys (from ``data``) return their declared type; unknown keys
    return the fallback ``type``, allowing the JS frontend to add
    ``library_1``, ``library_2``, … at runtime.
    """

    def __init__(self, fallback_type, data=None):
        super().__init__()
        self._fallback_type = fallback_type
        self._data = data or {}

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        return (self._fallback_type,)

    def __contains__(self, key):
        return True


class Prompt808LibrarySelect:
    """ComfyUI node for selecting multiple libraries."""

    CATEGORY = "Prompt808"
    DESCRIPTION = (
        "Select multiple libraries for combined prompt generation. "
        "Connect to the Prompt808 Generate node's 'libraries' input."
    )
    FUNCTION = "select"
    RETURN_TYPES = ("P808_LIBRARIES",)
    RETURN_NAMES = ("libraries",)
    OUTPUT_TOOLTIPS = ("Selected library names for multi-library generation",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _FlexibleInputType(_any_type),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def select(self, **kwargs):
        """Collect enabled library names from dynamic slot inputs."""
        libraries = []
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            if not key.upper().startswith("LIBRARY_"):
                continue
            if not isinstance(value, dict):
                continue
            if value.get("on") and value.get("name"):
                name = value["name"]
                if name not in libraries:
                    libraries.append(name)

        if not libraries:
            log.warning("No libraries enabled in Library Select node")

        return (libraries,)
