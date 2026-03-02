"""Tag vocabulary store — CRUD backed by SQLite.

Maintains a growing vocabulary of normalized tags with canonical forms.
When a new tag's embedding is >0.85 similar to an existing tag, it maps
to the existing canonical form rather than creating a new tag.

Vocabulary entry schema (returned as dict):
{
    "canonical": "golden_hour",
    "aliases": ["golden hour", "golden hour lighting"],
    "count": 12
}
"""

import json
import logging

from ..core import database, library_manager

log = logging.getLogger("prompt808.store.vocabulary")


def get_all():
    """Return the full vocabulary dict: {canonical: {canonical, aliases, count}}."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT canonical, aliases, count FROM vocabulary WHERE library_id=?",
        (lib_id,)
    ).fetchall()

    result = {}
    for r in rows:
        aliases = json.loads(r["aliases"]) if r["aliases"] else []
        result[r["canonical"]] = {
            "canonical": r["canonical"],
            "aliases": aliases,
            "count": r["count"],
        }
    return result


def clear_all():
    """Delete all vocabulary entries. Returns count removed."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        n = db.execute(
            "SELECT COUNT(*) as cnt FROM vocabulary WHERE library_id=?", (lib_id,)
        ).fetchone()["cnt"]
        db.execute("DELETE FROM vocabulary WHERE library_id=?", (lib_id,))
        db.commit()
    return n


def get_canonical(tag):
    """Return the canonical form of a tag, or the tag itself if not found."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()

    # Check if this tag is a canonical form
    row = db.execute(
        "SELECT canonical FROM vocabulary WHERE library_id=? AND canonical=?",
        (lib_id, tag)
    ).fetchone()
    if row:
        return tag

    # Check if this tag is an alias — search all rows
    rows = db.execute(
        "SELECT canonical, aliases FROM vocabulary WHERE library_id=?",
        (lib_id,)
    ).fetchall()
    for r in rows:
        aliases = json.loads(r["aliases"]) if r["aliases"] else []
        if tag in aliases:
            return r["canonical"]

    return tag


def add_tag(tag, canonical=None):
    """Register a tag. If canonical is provided, tag becomes an alias.

    Returns the canonical form.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        if canonical and canonical != tag:
            # Add as alias to existing canonical form
            row = db.execute(
                "SELECT aliases, count FROM vocabulary WHERE library_id=? AND canonical=?",
                (lib_id, canonical)
            ).fetchone()
            if row:
                aliases = json.loads(row["aliases"]) if row["aliases"] else []
                if tag not in aliases:
                    aliases.append(tag)
                db.execute(
                    "UPDATE vocabulary SET aliases=?, count=? WHERE library_id=? AND canonical=?",
                    (json.dumps(aliases), row["count"] + 1, lib_id, canonical)
                )
            else:
                db.execute(
                    "INSERT INTO vocabulary (canonical, library_id, aliases, count) VALUES (?,?,?,?)",
                    (canonical, lib_id, json.dumps([tag]), 1)
                )
            db.commit()
            return canonical

        # Register as new canonical form or increment existing
        row = db.execute(
            "SELECT count FROM vocabulary WHERE library_id=? AND canonical=?",
            (lib_id, tag)
        ).fetchone()
        if row:
            db.execute(
                "UPDATE vocabulary SET count=? WHERE library_id=? AND canonical=?",
                (row["count"] + 1, lib_id, tag)
            )
        else:
            db.execute(
                "INSERT INTO vocabulary (canonical, library_id, aliases, count) VALUES (?,?,?,?)",
                (tag, lib_id, json.dumps([]), 1)
            )
        db.commit()
        return tag


def add_tags(tags):
    """Register multiple tags at once (all as new canonical forms)."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        for tag in tags:
            row = db.execute(
                "SELECT count FROM vocabulary WHERE library_id=? AND canonical=?",
                (lib_id, tag)
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE vocabulary SET count=? WHERE library_id=? AND canonical=?",
                    (row["count"] + 1, lib_id, tag)
                )
            else:
                db.execute(
                    "INSERT INTO vocabulary (canonical, library_id, aliases, count) VALUES (?,?,?,?)",
                    (tag, lib_id, json.dumps([]), 1)
                )
        db.commit()


def get_all_canonical_tags():
    """Return sorted list of all canonical tag forms."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT canonical FROM vocabulary WHERE library_id=? ORDER BY canonical",
        (lib_id,)
    ).fetchall()
    return [r["canonical"] for r in rows]


def count():
    """Return total number of canonical tags."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    return db.execute(
        "SELECT COUNT(*) as cnt FROM vocabulary WHERE library_id=?", (lib_id,)
    ).fetchone()["cnt"]
