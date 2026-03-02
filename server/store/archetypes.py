"""Archetype store — CRUD backed by SQLite.

Archetypes are auto-generated groups of compatible elements discovered
by DBSCAN clustering + LLM naming. Each archetype defines which element
tags are compatible, enabling genre-aware filtering.

Archetype schema:
{
    "id": "mountain_dramatic",
    "name": "Dramatic Mountain Sunset",
    "compatible": {
        "environment_tags": ["mountain", "alpine"],
        "lighting_tags": ["dramatic", "golden_hour"],
    },
    "element_ids": ["elem_1", "elem_2", ...],
    "negative_hints": ["indoor", "studio", "portrait"],
    "generated": "2026-02-21"
}
"""

import json
import logging
from datetime import date

from ..core import database, library_manager

log = logging.getLogger("prompt808.store.archetypes")


def _row_to_dict(row):
    """Convert a sqlite3.Row to an archetype dict, parsing JSON columns."""
    d = dict(row)
    d["compatible"] = json.loads(d["compatible"]) if d.get("compatible") else {}
    d["element_ids"] = json.loads(d["element_ids"]) if d.get("element_ids") else []
    d["negative_hints"] = json.loads(d["negative_hints"]) if d.get("negative_hints") else []
    d.pop("library_id", None)
    return d


def get_all():
    """Return all archetypes."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT * FROM archetypes WHERE library_id=?", (lib_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_names():
    """Return list of archetype display names."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT name, id FROM archetypes WHERE library_id=?", (lib_id,)
    ).fetchall()
    return [r["name"] or r["id"] or "unknown" for r in rows]


def get_by_id(archetype_id):
    """Return a single archetype by ID, or None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT * FROM archetypes WHERE library_id=? AND id=?",
        (lib_id, archetype_id)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_by_name(name):
    """Return a single archetype by display name, or None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT * FROM archetypes WHERE library_id=? AND name=?",
        (lib_id, name)
    ).fetchone()
    return _row_to_dict(row) if row else None


def replace_all(archetypes):
    """Replace the entire archetypes list (used after regeneration)."""
    today = date.today().isoformat()
    for a in archetypes:
        if "generated" not in a:
            a["generated"] = today

    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        db.execute("DELETE FROM archetypes WHERE library_id=?", (lib_id,))
        db.executemany(
            """INSERT INTO archetypes
               (id, library_id, name, compatible, element_ids,
                negative_hints, generated)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    a.get("id"),
                    lib_id,
                    a.get("name"),
                    json.dumps(a.get("compatible", {})),
                    json.dumps(a.get("element_ids", [])),
                    json.dumps(a.get("negative_hints", [])),
                    a.get("generated"),
                )
                for a in archetypes
            ]
        )
        db.commit()
    log.info("Archetypes replaced: %d total", len(archetypes))


def clear_all():
    """Delete all archetypes. Returns count removed."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        n = db.execute(
            "SELECT COUNT(*) as cnt FROM archetypes WHERE library_id=?", (lib_id,)
        ).fetchone()["cnt"]
        db.execute("DELETE FROM archetypes WHERE library_id=?", (lib_id,))
        db.commit()
    return n


def delete(archetype_id):
    """Delete an archetype by ID. Returns True if found and deleted."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        cursor = db.execute(
            "DELETE FROM archetypes WHERE library_id=? AND id=?",
            (lib_id, archetype_id)
        )
        db.commit()
    return cursor.rowcount > 0


def count():
    """Return total number of archetypes."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    return db.execute(
        "SELECT COUNT(*) as cnt FROM archetypes WHERE library_id=?", (lib_id,)
    ).fetchone()["cnt"]
