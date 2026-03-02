"""Library management API — CRUD for elements and archetypes.

Provides endpoints for browsing, editing, and deleting elements
and archetypes in the user's library.
"""

import logging
from pathlib import Path

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel
except ImportError:
    APIRouter = None

    class BaseModel:
        """Stub so model classes can still be defined."""

from ..core import library_manager

log = logging.getLogger("prompt808.api.library")

if APIRouter is not None:
    router = APIRouter()
else:
    class _NoOpRouter:
        """Stub router whose decorators are identity functions."""
        def _noop(self, *a, **kw):
            return lambda fn: fn
        get = post = put = patch = delete = _noop

    router = _NoOpRouter()


# --- Element endpoints ---


class ElementResponse(BaseModel):
    id: str
    desc: str
    category: str
    tags: list[str]
    attributes: dict = {}
    source_photo: str | None = None
    subject_type: str | None = None
    thumbnail: str | None = None
    added: str | None = None


@router.get("/elements")
async def list_elements(
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List elements with optional category filter and pagination."""
    from ..store import elements

    if category:
        all_elems = elements.get_by_category(category)
    else:
        all_elems = elements.get_all()

    total = len(all_elems)
    page = all_elems[offset:offset + limit]

    return {
        "elements": page,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/elements/{element_id}")
async def get_element(element_id: str):
    """Get a single element by ID."""
    from ..store import elements

    elem = elements.get_by_id(element_id)
    if not elem:
        raise HTTPException(404, f"Element '{element_id}' not found")
    return elem


@router.delete("/elements/{element_id}")
async def delete_element(element_id: str):
    """Delete an element by ID."""
    from ..store import elements

    if not elements.delete(element_id):
        raise HTTPException(404, f"Element '{element_id}' not found")

    # Invalidate prompt cache
    try:
        from ..core import prompt_cache
        prompt_cache.invalidate()
    except Exception:
        pass

    return {"status": "deleted", "id": element_id}


class ElementUpdate(BaseModel):
    desc: str | None = None
    tags: list[str] | None = None
    attributes: dict | None = None


@router.patch("/elements/{element_id}")
async def update_element(element_id: str, update: ElementUpdate):
    """Update fields on an element."""
    from ..store import elements

    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")

    result = elements.update(element_id, updates)
    if not result:
        raise HTTPException(404, f"Element '{element_id}' not found")

    # Invalidate prompt cache
    try:
        from ..core import prompt_cache
        prompt_cache.invalidate()
    except Exception:
        pass

    return result


@router.get("/categories")
async def list_categories():
    """List all element categories with counts."""
    from ..store import elements

    return {
        "categories": elements.get_categories(),
        "counts": elements.count_by_category(),
    }


# --- Archetype endpoints ---


@router.get("/archetypes")
async def list_archetypes():
    """List all archetypes."""
    from ..store import archetypes

    return {"archetypes": archetypes.get_all()}


@router.get("/archetypes/{archetype_id}")
async def get_archetype(archetype_id: str):
    """Get a single archetype by ID."""
    from ..store import archetypes

    arch = archetypes.get_by_id(archetype_id)
    if not arch:
        raise HTTPException(404, f"Archetype '{archetype_id}' not found")
    return arch


@router.delete("/archetypes/{archetype_id}")
async def delete_archetype(archetype_id: str):
    """Delete an archetype by ID."""
    from ..store import archetypes

    if not archetypes.delete(archetype_id):
        raise HTTPException(404, f"Archetype '{archetype_id}' not found")
    return {"status": "deleted", "id": archetype_id}


@router.post("/archetypes/regenerate")
async def regenerate_archetypes():
    """Force regeneration of all archetypes from the current library."""
    from ..core import archetypes as archetype_gen
    from ..core import model_manager
    from ..store import archetypes as archetype_store
    from ..store import elements

    all_elements = elements.get_all()
    if not all_elements:
        raise HTTPException(400, "Library is empty — add elements first")

    new_archetypes = archetype_gen.generate_archetypes(
        all_elements,
        use_llm_naming=True,
        model_manager=model_manager,
    )
    archetype_store.replace_all(new_archetypes)

    return {
        "status": "regenerated",
        "count": len(new_archetypes),
        "archetypes": [a.get("name") for a in new_archetypes],
    }


# --- Photos (grouped by thumbnail) ---


@router.get("/photos")
async def list_photos():
    """List all analyzed photos with their associated element counts.

    Groups elements by thumbnail and returns one entry per photo.
    """
    from ..store import elements

    all_elements = elements.get_all()

    # Group elements by thumbnail
    photos = {}
    for elem in all_elements:
        thumb = elem.get("thumbnail")
        if not thumb:
            continue
        if thumb not in photos:
            photos[thumb] = {
                "thumbnail": thumb,
                "source_photo": elem.get("source_photo"),
                "subject_type": elem.get("subject_type"),
                "added": elem.get("added"),
                "element_count": 0,
                "categories": [],
            }
        photos[thumb]["element_count"] += 1
        cat = elem.get("category")
        if cat and cat not in photos[thumb]["categories"]:
            photos[thumb]["categories"].append(cat)

    # Sort by most recent first
    result = sorted(photos.values(), key=lambda p: p.get("added") or "", reverse=True)
    return {"photos": result, "total": len(result)}


@router.get("/photos/{thumbnail}/elements")
async def get_photo_elements(thumbnail: str):
    """Get all elements associated with a specific photo thumbnail."""
    from ..store import elements

    all_elements = elements.get_all()
    photo_elements = [e for e in all_elements if e.get("thumbnail") == thumbnail]
    if not photo_elements:
        raise HTTPException(404, f"No elements found for thumbnail '{thumbnail}'")
    return {"elements": photo_elements, "total": len(photo_elements)}


@router.delete("/photos/{thumbnail}")
async def delete_photo(thumbnail: str):
    """Delete a photo and all its associated elements.

    Removes: the thumbnail file, all elements linked to it,
    and the image embedding cache entry.
    """
    from ..core import image_embeddings, prompt_cache
    from ..store import elements

    removed_count = elements.delete_by_thumbnail(thumbnail)
    if removed_count == 0:
        raise HTTPException(404, f"No elements found for thumbnail '{thumbnail}'")

    # Delete the thumbnail file
    thumb_path = library_manager.get_thumbnails_dir() / thumbnail
    if thumb_path.is_file():
        thumb_path.unlink()

    # Remove image embedding cache entry by hash prefix from thumbnail name
    # Thumbnail format: {stem}_{hash8}.jpg -> extract hash8
    stem = Path(thumbnail).stem  # e.g. "photo_abc12345"
    hash_prefix = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    img_emb_removed = 0
    if hash_prefix:
        img_emb_removed = image_embeddings.remove_by_hash_prefix(hash_prefix)

    # Invalidate prompt cache since library changed
    try:
        prompt_cache.invalidate()
    except Exception:
        pass

    log.info("Photo deleted: %s (%d elements removed)", thumbnail, removed_count)
    return {
        "status": "deleted",
        "thumbnail": thumbnail,
        "elements_removed": removed_count,
        "image_embeddings_removed": img_emb_removed,
    }


# --- Library stats ---


@router.get("/stats")
async def library_stats():
    """Return library statistics."""
    from ..core import prompt_cache
    from ..store import archetypes, elements, vocabulary

    return {
        "elements": elements.count(),
        "categories": len(elements.get_categories()),
        "category_counts": elements.count_by_category(),
        "archetypes": archetypes.count(),
        "vocabulary_size": vocabulary.count(),
        "cached_prompts": prompt_cache.size(),
        "library_version": elements.get_library_version(),
    }


# --- Reset all data ---


@router.delete("/reset")
async def reset_all_data():
    """Clear all library data: elements, archetypes, vocabulary, style profiles, prompt cache, and embeddings cache."""
    from ..core import embeddings, image_embeddings, prompt_cache, style_profile
    from ..store import archetypes, elements, vocabulary

    el = elements.clear_all()
    ar = archetypes.clear_all()
    vo = vocabulary.clear_all()
    em = embeddings.clear_cache()
    im = image_embeddings.clear_cache()
    style_profile.reset()
    prompt_cache.invalidate()

    # Clear thumbnails
    thumbnails_dir = library_manager.get_thumbnails_dir()
    thumb_count = 0
    if thumbnails_dir.is_dir():
        for f in thumbnails_dir.iterdir():
            if f.is_file():
                f.unlink()
                thumb_count += 1

    log.warning("All data reset: %d elements, %d archetypes, %d vocab tags, %d embeddings, %d image embeddings, %d thumbnails cleared",
                el, ar, vo, em, im, thumb_count)

    return {
        "status": "all_data_cleared",
        "elements_removed": el,
        "archetypes_removed": ar,
        "vocabulary_removed": vo,
        "embeddings_removed": em,
        "image_embeddings_removed": im,
        "thumbnails_removed": thumb_count,
    }
