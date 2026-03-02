"""Prompt generation API — SSE endpoint for ComfyUI bridge node.

GET /api/generate — SSE stream that sends progress events and final prompt.
POST /api/generate — synchronous single-prompt generation.
"""

import asyncio
import json
import logging
import uuid

try:
    from fastapi import APIRouter, Query
    from pydantic import BaseModel
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    APIRouter = None

    class BaseModel:
        """Stub so model classes can still be defined."""

log = logging.getLogger("prompt808.api.generation")

if APIRouter is not None:
    router = APIRouter()
else:
    class _NoOpRouter:
        """Stub router whose decorators are identity functions."""
        def _noop(self, *a, **kw):
            return lambda fn: fn
        get = post = put = patch = delete = _noop

    router = _NoOpRouter()


class GenerateRequest(BaseModel):
    seed: int
    archetype_id: str = "Any"
    style: str = "Any"
    mood: str = "Any"
    model_name: str = "None"
    quantization: str = "FP16"
    enrichment: str = "Vivid"
    keep_model_loaded: bool = True
    temperature: float = 0.9
    max_tokens: int = 1024
    prefix: str = ""
    suffix: str = ""
    batch_count: int = 1
    debug: bool = False


class GenerateResponse(BaseModel):
    prompt: str
    negative_prompt: str
    archetype_used: str
    elements_used: list[str]
    status: str
    seed: int


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    count: int


@router.post("/generate")
async def generate_sync(req: GenerateRequest):
    """Synchronous prompt generation. Supports batch_count > 1."""
    if req.batch_count <= 1:
        result = await asyncio.to_thread(_run_generation, req)
        return GenerateResponse(**result)
    else:
        results = []
        for i in range(req.batch_count):
            batch_req = req.model_copy(update={"seed": req.seed + i})
            result = await asyncio.to_thread(_run_generation, batch_req)
            results.append(GenerateResponse(**result))
        return BatchGenerateResponse(results=results, count=len(results))


@router.get("/generate")
async def generate_sse(
    seed: int = Query(...),
    archetype_id: str = Query("Any"),
    style: str = Query("Any"),
    mood: str = Query("Any"),
    model_name: str = Query("None"),
    quantization: str = Query("FP16"),
    enrichment: str = Query("Vivid"),
    keep_model_loaded: bool = Query(True),
    temperature: float = Query(0.9),
    max_tokens: int = Query(1024),
    prefix: str = Query(""),
    suffix: str = Query(""),
    debug: bool = Query(False),
):
    """SSE endpoint for ComfyUI bridge node.

    Streams events:
      - progress: {value: N, max: M, message: "..."}
      - result: {prompt: "...", negative_prompt: "...", ...}
      - error: {message: "..."}
    """
    req = GenerateRequest(
        seed=seed,
        archetype_id=archetype_id,
        style=style,
        mood=mood,
        model_name=model_name,
        quantization=quantization,
        enrichment=enrichment,
        keep_model_loaded=keep_model_loaded,
        temperature=temperature,
        max_tokens=max_tokens,
        prefix=prefix,
        suffix=suffix,
        debug=debug,
    )

    generation_id = str(uuid.uuid4())[:8]

    async def event_generator():
        try:
            # Progress: starting
            yield {
                "event": "progress",
                "data": json.dumps({
                    "id": generation_id,
                    "value": 0,
                    "max": 4,
                    "message": "Starting generation...",
                }),
            }

            # Progress: checking cache
            yield {
                "event": "progress",
                "data": json.dumps({
                    "id": generation_id,
                    "value": 1,
                    "max": 4,
                    "message": "Checking prompt cache...",
                }),
            }

            # Run generation in thread pool
            result = await asyncio.to_thread(_run_generation, req)

            # Progress: composing
            yield {
                "event": "progress",
                "data": json.dumps({
                    "id": generation_id,
                    "value": 3,
                    "max": 4,
                    "message": "Prompt composed",
                }),
            }

            # Final result
            yield {
                "event": "result",
                "data": json.dumps({
                    "id": generation_id,
                    "prompt": result["prompt"],
                    "negative_prompt": result["negative_prompt"],
                    "archetype_used": result["archetype_used"],
                    "elements_used": result["elements_used"],
                    "status": result["status"],
                    "seed": result["seed"],
                }),
            }

            # Progress: complete
            yield {
                "event": "progress",
                "data": json.dumps({
                    "id": generation_id,
                    "value": 4,
                    "max": 4,
                    "message": "Complete",
                }),
            }

        except Exception as e:
            log.error("SSE generation failed: %s", e, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "id": generation_id,
                    "message": str(e),
                }),
            }

    return EventSourceResponse(event_generator())


def _run_generation(req: GenerateRequest) -> dict:
    """Execute prompt generation (runs in thread pool)."""
    from ..core import generator, model_manager, style_profile
    from ..store import archetypes, elements

    result = generator.generate_prompt(
        seed=req.seed,
        archetype_id=req.archetype_id,
        style=req.style,
        mood=req.mood,
        model_name=req.model_name,
        quantization=req.quantization,
        enrichment=req.enrichment,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        model_manager=model_manager,
        element_store=elements,
        archetype_store=archetypes,
        style_profile_module=style_profile,
        debug=req.debug,
    )

    # Include the seed used in the response
    result["seed"] = req.seed

    # Apply prefix/suffix
    if result.get("prompt") and (req.prefix or req.suffix):
        parts = []
        if req.prefix:
            parts.append(req.prefix.strip())
        parts.append(result["prompt"])
        if req.suffix:
            parts.append(req.suffix.strip())
        result["prompt"] = " ".join(parts)

    # Post-generation model lifecycle
    if req.model_name and req.model_name != "None":
        if not req.keep_model_loaded:
            model_manager.unload_model()
        else:
            model_manager.offload_model()

    return result


@router.get("/generate/options")
async def generation_options():
    """Return available options for the generation UI."""
    from ..core.generator import get_available_moods, get_available_styles
    from ..core.model_manager import get_model_names
    from ..store import archetypes

    return {
        "prompt_types": get_available_styles(),
        "moods": get_available_moods(),
        "archetypes": ["Any"] + archetypes.get_names(),
        "models": get_model_names(),
    }
