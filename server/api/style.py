"""Style profile API — view and manage learned photographer style.

Provides endpoints for viewing per-genre style profiles, getting
style context for generation, and resetting profiles.
"""

import logging

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    APIRouter = None

log = logging.getLogger("prompt808.api.style")

if APIRouter is not None:
    router = APIRouter()
else:
    class _NoOpRouter:
        """Stub router whose decorators are identity functions."""
        def _noop(self, *a, **kw):
            return lambda fn: fn
        get = post = put = patch = delete = _noop

    router = _NoOpRouter()


@router.get("/style/profiles")
async def list_profiles():
    """List all genre style profiles with summaries."""
    from ..core import style_profile

    return {
        "genres": style_profile.get_all_genres(),
        "summary": style_profile.get_summary(),
    }


@router.get("/style/profile/{genre}")
async def get_profile(genre: str):
    """Get detailed style profile for a genre."""
    from ..core import style_profile

    # Check the genre exists at all (even with few observations)
    all_genres = style_profile.get_all_genres()
    if genre not in all_genres:
        raise HTTPException(404, f"No style profile for genre '{genre}'")

    profile = style_profile.get_genre_profile(genre)
    context = style_profile.get_style_context(genre)
    return {
        "genre": genre,
        "profile": profile,
        "context_text": context,
    }


@router.delete("/style/profile/{genre}")
async def reset_genre_profile(genre: str):
    """Reset style profile for a specific genre."""
    from ..core import style_profile

    style_profile.reset(genre)
    return {"status": "reset", "genre": genre}


@router.delete("/style/profiles")
async def reset_all_profiles():
    """Reset all style profiles."""
    from ..core import style_profile

    style_profile.reset()
    return {"status": "all_profiles_reset"}
