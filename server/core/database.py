"""SQLite database layer for Prompt808.

Single-file database at user_data/prompt808.db. Provides:
- WAL journal mode for concurrent reads + serialized writes
- Module-level connection singleton with check_same_thread=False
- Single threading.Lock for all writes
- Schema auto-creation on first get_db() call
"""

import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger("prompt808.database")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "user_data"

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()

# Test override — monkeypatched by test fixtures
_db_path_override: str | None = None

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS libraries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    is_active   INTEGER DEFAULT 0,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    id           TEXT    NOT NULL,
    library_id   INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    desc         TEXT,
    category     TEXT,
    tags         TEXT,
    attributes   TEXT,
    source_photo TEXT,
    subject_type TEXT,
    thumbnail    TEXT,
    medium       TEXT,
    is_photograph INTEGER,
    extraction_type TEXT,
    added        TEXT,
    PRIMARY KEY (library_id, id)
);
CREATE INDEX IF NOT EXISTS idx_elements_category ON elements(library_id, category);
CREATE INDEX IF NOT EXISTS idx_elements_thumbnail ON elements(library_id, thumbnail);

CREATE TABLE IF NOT EXISTS archetypes (
    id              TEXT    NOT NULL,
    library_id      INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    name            TEXT,
    compatible      TEXT,
    element_ids     TEXT,
    negative_hints  TEXT,
    generated       TEXT,
    PRIMARY KEY (library_id, id)
);

CREATE TABLE IF NOT EXISTS vocabulary (
    canonical    TEXT    NOT NULL,
    library_id   INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    aliases      TEXT,
    count        INTEGER DEFAULT 1,
    PRIMARY KEY (library_id, canonical)
);

CREATE TABLE IF NOT EXISTS style_profiles (
    genre        TEXT    NOT NULL,
    library_id   INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    observations INTEGER DEFAULT 0,
    patterns     TEXT,
    last_updated TEXT,
    PRIMARY KEY (library_id, genre)
);

CREATE TABLE IF NOT EXISTS embeddings_cache (
    element_id   TEXT    NOT NULL,
    library_id   INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    embedding    BLOB    NOT NULL,
    PRIMARY KEY (library_id, element_id)
);

CREATE TABLE IF NOT EXISTS image_embeddings_cache (
    content_hash TEXT    NOT NULL,
    library_id   INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    embedding    BLOB    NOT NULL,
    PRIMARY KEY (library_id, content_hash)
);

CREATE TABLE IF NOT EXISTS prompt_cache (
    cache_key       TEXT NOT NULL,
    library_id      INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    prompt          TEXT,
    negative_prompt TEXT,
    seed            INTEGER,
    archetype_id    TEXT,
    style           TEXT,
    PRIMARY KEY (library_id, cache_key)
);

CREATE TABLE IF NOT EXISTS generate_settings (
    key   TEXT PRIMARY KEY DEFAULT 'default',
    value TEXT
);
"""


def _get_db_path() -> Path:
    if _db_path_override is not None:
        return Path(_db_path_override)
    return _BASE_DIR / "prompt808.db"


def get_db() -> sqlite3.Connection:
    """Return the module-level SQLite connection, creating DB + schema if needed."""
    global _conn
    if _conn is not None:
        return _conn

    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA foreign_keys=ON")
    _conn.executescript(_SCHEMA_SQL)
    _conn.commit()

    log.info("Database opened: %s", db_path)
    return _conn


def write_lock() -> threading.Lock:
    """Return the global write lock for all DB mutations."""
    return _lock


def close():
    """Close the connection. Used for test teardown."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None
