"""Prompt808 Generate Node — ComfyUI native prompt generator.

Calls the generation pipeline directly via Python. Generation settings
are exposed as node dropdown inputs (library, prompt_type, archetype,
mood, llm_model, enrichment, quantization) so they are visible in the
workflow graph. Non-dropdown settings (temperature, max_tokens,
keep_model_loaded, debug) are still read from the sidebar SQLite settings.
"""

import json
import logging

log = logging.getLogger("prompt808.node")

_SETTINGS_DEFAULTS = {
    "prompt_type": "Any",
    "archetype": "Any",
    "mood": "Any",
    "llm_model": "None",
    "quantization": "FP16",
    "enrichment": "Vivid",
    "temperature": 0.9,
    "max_tokens": 1024,
    "keep_model_loaded": False,
    "debug": False,
}

PROMPT_TYPES = ["Any", "Architectural", "Boudoir", "Cinematic", "Documentary", "Erotica",
                "Fashion", "Fine Art", "Native", "Portrait", "Street"]
MOODS = ["Any", "Dramatic", "Elegant", "Ethereal", "Gritty", "Melancholic",
         "Mysterious", "Provocative", "Romantic", "Sensual", "Serene"]
QUANTIZATIONS = ["FP16", "FP8", "8-bit", "4-bit"]
ENRICHMENTS = ["Baseline", "Vivid", "Expressive", "Poetic", "Lyrical", "Freeform"]


class Prompt808Generate:
    """ComfyUI node that generates prompts via Prompt808."""

    CATEGORY = "Prompt808"
    DESCRIPTION = "Generates prompts from your Prompt808 library. Core settings are exposed as node inputs; temperature, max_tokens, and keep_model_loaded are configured in the sidebar. Debug mode is in ComfyUI Settings > Prompt808."
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "status")
    OUTPUT_TOOLTIPS = (
        "Generated prompt",
        "Negative prompt (terms to avoid)",
        "Generation status and archetype used",
    )

    @classmethod
    def INPUT_TYPES(cls):
        library_list = ["(active)"]
        archetype_list = ["Any"]
        model_list = ["None"]

        try:
            from .server.core import library_manager
            libs = library_manager.list_libraries()
        except Exception:
            try:
                from server.core import library_manager
                libs = library_manager.list_libraries()
            except Exception:
                libs = []

        if libs:
            library_list = ["(active)"] + [lib["name"] for lib in libs]
        else:
            library_list = ["(no libraries)"]

        try:
            from .server.store import archetypes as arch_store
            archetype_list = ["Any"] + arch_store.get_names()
        except Exception:
            try:
                from server.store import archetypes as arch_store
                archetype_list = ["Any"] + arch_store.get_names()
            except Exception:
                pass

        try:
            from .server.core.model_manager import get_model_names
            model_list = get_model_names()
        except Exception:
            try:
                from server.core.model_manager import get_model_names
                model_list = get_model_names()
            except Exception:
                pass

        # Read NSFW setting from database to filter adult content options
        nsfw = False
        try:
            try:
                from .server.core import database
            except ImportError:
                from server.core import database
            db = database.get_db()
            row = db.execute(
                "SELECT value FROM generate_settings WHERE key='app'"
            ).fetchone()
            if row and row["value"]:
                nsfw = json.loads(row["value"]).get("nsfw", False)
        except Exception:
            pass

        prompt_types = PROMPT_TYPES if nsfw else [t for t in PROMPT_TYPES if t not in ("Boudoir", "Erotica")]
        moods = MOODS if nsfw else [m for m in MOODS if m not in ("Sensual", "Provocative")]

        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for deterministic generation",
                }),
            },
            "optional": {
                "library": (library_list, {
                    "default": library_list[0],
                    "tooltip": "Library to generate from" if libs else "No libraries — open the Prompt808 sidebar (camera icon) to create one",
                }),
                "prompt_type": (prompt_types, {
                    "default": "Any",
                    "tooltip": "Prompt style (Cinematic, Documentary, etc.)",
                }),
                "archetype": (archetype_list, {
                    "default": "Any",
                    "tooltip": "Archetype to filter elements by",
                }),
                "mood": (moods, {
                    "default": "Any",
                    "tooltip": "Mood modifier for the generated prompt",
                }),
                "llm_model": (model_list, {
                    "default": model_list[0],
                    "tooltip": "LLM model for prompt composition (None = simple mode)",
                }),
                "enrichment": (ENRICHMENTS, {
                    "default": "Vivid",
                    "tooltip": "Creative enrichment level for LLM composition",
                }),
                "quantization": (QUANTIZATIONS, {
                    "default": "FP16",
                    "tooltip": "LLM quantization (FP16, FP8, 8-bit, 4-bit)",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Text prepended to the generated prompt (e.g. LoRA trigger word)",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Text appended to the generated prompt (e.g. quality tags)",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute — settings come from file

    def _load_settings(self):
        """Read generation settings persisted by the sidebar UI.

        Reads from the SQLite database (generate_settings table), which is
        where the sidebar saves settings via PUT /prompt808/api/generate/settings.

        Returns (settings_dict, used_defaults) where used_defaults is True
        when no settings were found or they failed to parse.
        """
        settings = dict(_SETTINGS_DEFAULTS)
        used_defaults = True
        try:
            try:
                from .server.core import database
            except ImportError:
                from server.core import database
            db = database.get_db()
            row = db.execute(
                "SELECT value FROM generate_settings WHERE key='default'"
            ).fetchone()
            if row and row["value"]:
                saved = json.loads(row["value"])
                settings.update(saved)
                used_defaults = False
        except Exception as e:
            log.warning("Failed to load saved settings: %s", e)
        return settings, used_defaults

    def generate(self, seed, library="(active)", prompt_type="Any",
                 archetype="Any", mood="Any", llm_model="None",
                 enrichment="Vivid", quantization="FP16",
                 prefix="", suffix=""):
        """Generate a prompt using node inputs + sidebar settings."""
        # Scope to the selected library (skip if using active library)
        token = None
        if library and library != "(active)":
            try:
                try:
                    from .server.core import library_manager
                except ImportError:
                    from server.core import library_manager
                token = library_manager._request_library.set(library)
            except Exception as e:
                log.warning("Failed to scope library '%s': %s", library, e)

        # Check if any library exists
        try:
            try:
                from .server.core import library_manager as _lm
            except ImportError:
                from server.core import library_manager as _lm
            if _lm.get_active() is None:
                msg = "No library exists \u2014 open the Prompt808 sidebar (camera icon) and create one first"
                log.warning(msg)
                return ("", "", msg)
        except Exception as e:
            log.warning("Failed to check library state: %s", e)

        try:
            # Read sidebar settings for non-dropdown fields only
            settings, used_defaults = self._load_settings()

            # Node inputs override sidebar settings for dropdown fields
            settings.update({
                "prompt_type": prompt_type,
                "archetype": archetype,
                "mood": mood,
                "llm_model": llm_model,
                "enrichment": enrichment,
                "quantization": quantization,
            })

            # Pre-flight: check element count
            try:
                try:
                    from .server.store import elements
                except ImportError:
                    from server.store import elements
                count = elements.count()
            except Exception as e:
                log.warning("Failed to check element count: %s", e)
                count = -1  # can't determine — let generation proceed

            if count == 0:
                msg = "Library is empty \u2014 open the Prompt808 sidebar and analyze some images first"
                log.warning(msg)
                prompt = prefix.strip() if prefix else ""
                return (prompt, "", msg)

            result = self._generate_native(seed, settings, prefix, suffix)

            # Build richer status line
            status_tag = result.get("status", "ok")
            if used_defaults:
                status_tag += " (default settings)"
            status_parts = [status_tag]
            status_parts.append(f"archetype: {result.get('archetype_used', 'unknown')}")
            status_parts.append(f"elements: {len(result.get('elements_used', []))}")
            status_parts.append(f"seed: {result.get('seed', seed)}")
            status = " | ".join(status_parts)

            return (result.get("prompt", ""), result.get("negative_prompt", ""), status)
        except Exception as e:
            error_msg = f"Prompt808 generation failed: {e}"
            log.error(error_msg, exc_info=True)
            return ("", "", f"ERROR: {error_msg}")
        finally:
            if token is not None:
                try:
                    try:
                        from .server.core import library_manager
                    except ImportError:
                        from server.core import library_manager
                    library_manager._request_library.reset(token)
                except Exception as e:
                    log.warning("Failed to reset library scope: %s", e)

    def _generate_native(self, seed, settings, prefix, suffix):
        """Direct Python call into the generation pipeline."""
        try:
            from .server.core import generator, model_manager, style_profile
            from .server.store import archetypes, elements
        except ImportError:
            from server.core import generator, model_manager, style_profile
            from server.store import archetypes, elements

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(4)
        except ImportError:
            pass

        if pbar:
            pbar.update_absolute(0, 4)

        llm_model = settings.get("llm_model", "None")
        keep = settings.get("keep_model_loaded", True)
        debug = settings.get("debug", False)

        result = generator.generate_prompt(
            seed=seed,
            archetype_id=settings.get("archetype", "Any"),
            style=settings.get("prompt_type", "Any"),
            mood=settings.get("mood", "Any"),
            model_name=llm_model,
            quantization=settings.get("quantization", "FP16"),
            enrichment=settings.get("enrichment", "Vivid"),
            temperature=float(settings.get("temperature", 0.9)),
            max_tokens=int(settings.get("max_tokens", 1024)),
            model_manager=model_manager,
            element_store=elements,
            archetype_store=archetypes,
            style_profile_module=style_profile,
            debug=debug,
        )
        result["seed"] = seed

        # Apply prefix/suffix (even if prompt is empty, so LoRA triggers etc. survive)
        if prefix or suffix:
            parts = []
            if prefix:
                parts.append(prefix.strip())
            if result.get("prompt"):
                parts.append(result["prompt"])
            if suffix:
                parts.append(suffix.strip())
            result["prompt"] = " ".join(parts)

        # Post-generation model lifecycle
        if llm_model and llm_model != "None":
            if not keep:
                model_manager.unload_model()
            else:
                model_manager.offload_model()

        if pbar:
            pbar.update_absolute(4, 4)

        # Broadcast result to sidebar (expected to fail outside ComfyUI)
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("prompt808.generation_result", {
                "prompt": result.get("prompt", ""),
                "negative_prompt": result.get("negative_prompt", ""),
                "archetype_used": result.get("archetype_used", ""),
                "elements_used": result.get("elements_used", []),
                "status": result.get("status", ""),
                "seed": result.get("seed", seed),
            })
        except Exception:
            log.debug("PromptServer broadcast unavailable")

        return result
