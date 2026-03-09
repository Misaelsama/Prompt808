"""Library export handler — builds a .p808 zip from SQLite data.

Queries all library-scoped tables, parses JSON-stored TEXT columns back
to native Python objects, and packages everything into an in-memory zip.
Optionally includes thumbnail image files.
"""

import io
import json
import logging
import zipfile
from datetime import datetime

log = logging.getLogger("prompt808.plugins.export.export")

# Tables to export and their JSON-encoded TEXT columns that need parsing.
# Embedding tables are skipped — BLOB data is regenerated on demand.
_EXPORT_TABLES = {
    "elements": {
        "json_columns": {"tags", "attributes"},
        "bool_columns": {"is_photograph"},
    },
    "archetypes": {
        "json_columns": {"compatible", "element_ids", "negative_hints"},
        "bool_columns": set(),
    },
    "vocabulary": {
        "json_columns": {"aliases"},
        "bool_columns": set(),
    },
    "style_profiles": {
        "json_columns": {"patterns"},
        "bool_columns": set(),
    },
}


def handle_export(library_name, include_thumbnails=True):
    """Export a library to .p808 zip bytes.

    Args:
        library_name: Name of the library to export.
        include_thumbnails: Whether to include thumbnail files.

    Returns:
        Bytes of the zip file, or a dict with error info.
    """
    from ...core import database, library_manager

    db = database.get_db()

    # Find the library ID
    row = db.execute(
        "SELECT id FROM libraries WHERE name=?", (library_name,)
    ).fetchone()
    if row is None:
        return {"status": "error", "message": f"Library '{library_name}' not found"}
    lib_id = row["id"]

    counts = {}
    table_data = {}

    for table_name, config in _EXPORT_TABLES.items():
        rows = db.execute(
            f"SELECT * FROM {table_name} WHERE library_id=?", (lib_id,)
        ).fetchall()

        parsed_rows = []
        for r in rows:
            d = dict(r)
            d.pop("library_id", None)
            d.pop("source_photo", None)
            d.pop("added", None)
            d.pop("generated", None)
            d.pop("count", None)
            d.pop("last_updated", None)

            # Parse JSON TEXT columns back to native objects
            for col in config["json_columns"]:
                if col in d and isinstance(d[col], str):
                    try:
                        d[col] = json.loads(d[col])
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Convert SQLite integer booleans back to Python bools
            for col in config["bool_columns"]:
                if col in d:
                    d[col] = bool(d[col])

            parsed_rows.append(d)

        table_data[table_name] = parsed_rows
        counts[table_name] = len(parsed_rows)

    # Integrity self-check: warn about dangling element_ids in archetypes
    element_ids_set = {e["id"] for e in table_data.get("elements", [])}
    for arch in table_data.get("archetypes", []):
        eids = arch.get("element_ids", [])
        if isinstance(eids, list):
            dangling = [eid for eid in eids if eid not in element_ids_set]
            if dangling:
                log.warning(
                    "Export: archetype '%s' has %d dangling element_ids: %s",
                    arch.get("name", arch.get("id", "?")),
                    len(dangling),
                    dangling[:5],
                )

    # Build metadata
    metadata = {
        "library_name": library_name,
        "exported_at": datetime.now().isoformat(),
        "format_version": 1,
        "counts": counts,
    }

    # Package into zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        for table_name, rows in table_data.items():
            if rows:
                zf.writestr(f"{table_name}.json", json.dumps(rows, indent=2))

        # Include thumbnails if requested
        if include_thumbnails:
            thumbs_dir = library_manager._LIBRARIES_DIR / library_name / "thumbnails"
            if thumbs_dir.is_dir():
                for f in thumbs_dir.iterdir():
                    if f.is_file():
                        zf.write(str(f), f"thumbnails/{f.name}")

    return buf.getvalue()
