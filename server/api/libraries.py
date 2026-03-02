"""Library management API — CRUD for multi-library support.

Provides endpoints for creating, listing, switching, renaming,
and deleting isolated photo libraries.
"""

import logging

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError:
    APIRouter = None

    class BaseModel:
        """Stub so model classes can still be defined."""

from ..core import library_manager

log = logging.getLogger("prompt808.api.libraries")

if APIRouter is not None:
    router = APIRouter()
else:
    class _NoOpRouter:
        """Stub router whose decorators are identity functions."""
        def _noop(self, *a, **kw):
            return lambda fn: fn
        get = post = put = patch = delete = _noop

    router = _NoOpRouter()


class LibraryCreate(BaseModel):
    name: str


class LibrarySwitch(BaseModel):
    name: str


class LibraryRename(BaseModel):
    name: str


@router.get("/libraries")
async def list_libraries():
    """List all libraries with active flag and stats."""
    return {"libraries": library_manager.list_libraries()}


@router.post("/libraries", status_code=201)
async def create_library(body: LibraryCreate):
    """Create a new library."""
    try:
        name = library_manager.create_library(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "created", "name": name}


@router.put("/libraries/active")
async def switch_library(body: LibrarySwitch):
    """Switch the active library."""
    try:
        library_manager.set_active(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "switched", "active": library_manager.get_active()}


@router.patch("/libraries/{name}")
async def rename_library(name: str, body: LibraryRename):
    """Rename a library."""
    try:
        new_name = library_manager.rename_library(name, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "renamed", "old_name": name, "new_name": new_name}


@router.delete("/libraries/{name}")
async def delete_library(name: str):
    """Delete a library. Refuses if it's the last remaining library."""
    try:
        library_manager.delete_library(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "deleted", "name": name}
