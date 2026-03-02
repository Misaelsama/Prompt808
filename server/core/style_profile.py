"""Per-genre style profiling from analyzed photos.

Tracks the user's aesthetic style across multiple analyses, building
a compositional DNA profile per genre. Used as context during prompt
generation to bias output toward the user's aesthetic.

Profiles can be rebuilt from current library elements at any time,
ensuring they stay in sync after photo or element deletions.

Backed by SQLite.
"""

import json
import logging
from collections import defaultdict
from datetime import date

from . import database, library_manager

log = logging.getLogger("prompt808.style_profile")

# Minimum observations before a pattern is considered stable
MIN_OBSERVATIONS = 3


def update_from_analysis(analysis_result):
    """Update style profile from a photo analysis result.

    Extracts style signals from the analyzed elements and records them
    under the appropriate genre.

    Args:
        analysis_result: Output from analyzer.analyze_photo(), must have
                        'subject_type' and 'elements'.
    """
    subject_type = analysis_result.get("subject_type", "unknown") or "unknown"
    elements = analysis_result.get("elements", [])
    if not elements:
        return

    signals = _extract_style_signals(elements)

    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        row = db.execute(
            "SELECT observations, patterns FROM style_profiles WHERE library_id=? AND genre=?",
            (lib_id, subject_type)
        ).fetchone()

        if row:
            observations = row["observations"] + 1
            patterns = json.loads(row["patterns"]) if row["patterns"] else {}
        else:
            observations = 1
            patterns = {}

        today = date.today().isoformat()

        # Merge signals into patterns with recency weighting
        for dimension, values in signals.items():
            if dimension not in patterns:
                patterns[dimension] = {}
            for value, weight in values.items():
                current = patterns[dimension].get(value, 0.0)
                alpha = 1.0 / observations
                patterns[dimension][value] = current * (1 - alpha) + weight * alpha

        db.execute(
            """INSERT OR REPLACE INTO style_profiles
               (genre, library_id, observations, patterns, last_updated)
               VALUES (?,?,?,?,?)""",
            (subject_type, lib_id, observations, json.dumps(patterns), today)
        )
        db.commit()

    log.info("Style profile updated for genre '%s' (observation #%d)",
             subject_type, observations)


def get_genre_profile(genre):
    """Get style profile for a specific genre.

    Returns dict of {dimension: {value: score}} sorted by score descending,
    or empty dict if no profile exists.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT observations, patterns FROM style_profiles WHERE library_id=? AND genre=?",
        (lib_id, genre)
    ).fetchone()

    if not row or row["observations"] < MIN_OBSERVATIONS:
        return {}

    patterns = json.loads(row["patterns"]) if row["patterns"] else {}
    result = {}
    for dimension, values in patterns.items():
        sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        result[dimension] = sorted_values

    return result


def get_style_context(genre, max_traits=5):
    """Generate a natural language style context for prompt generation.

    Args:
        genre: Subject type to get style for.
        max_traits: Maximum traits per dimension to include.

    Returns:
        String describing the photographer's style, or empty string if
        insufficient data.
    """
    profile = get_genre_profile(genre)
    if not profile:
        return ""

    parts = []
    dimension_labels = {
        "lighting": "Lighting preference",
        "composition": "Composition style",
        "palette": "Color palette",
        "mood": "Mood tendency",
        "camera": "Camera approach",
        "technique": "Technique",
        "environment": "Setting preference",
    }

    for dimension, values in profile.items():
        label = dimension_labels.get(dimension, dimension.replace("_", " ").title())
        top_values = list(values.items())[:max_traits]
        if top_values:
            significant = [(v, s) for v, s in top_values if s > 0.1]
            if significant:
                trait_strs = [v.replace("_", " ") for v, _ in significant]
                parts.append(f"- {label}: {', '.join(trait_strs)}")

    if not parts:
        return ""

    return "Style profile:\n" + "\n".join(parts)


def get_all_genres():
    """Return list of genre names that have style profiles."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT genre FROM style_profiles WHERE library_id=? ORDER BY genre",
        (lib_id,)
    ).fetchall()
    return [r["genre"] for r in rows]


def get_summary():
    """Return a summary of all style profiles."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT genre, observations, patterns, last_updated FROM style_profiles WHERE library_id=?",
        (lib_id,)
    ).fetchall()

    summary = {}
    for row in rows:
        genre = row["genre"]
        obs = row["observations"]
        patterns = json.loads(row["patterns"]) if row["patterns"] else {}
        top_patterns = {}
        for dim, values in patterns.items():
            if values:
                top = max(values, key=values.get)
                top_patterns[dim] = top
        summary[genre] = {
            "observations": obs,
            "last_updated": row["last_updated"],
            "top_patterns": top_patterns,
        }

    return summary


def reset(genre=None):
    """Reset style profile for a genre, or all profiles if genre is None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        if genre is None:
            db.execute("DELETE FROM style_profiles WHERE library_id=?", (lib_id,))
            log.info("All style profiles reset")
        else:
            db.execute(
                "DELETE FROM style_profiles WHERE library_id=? AND genre=?",
                (lib_id, genre)
            )
            log.info("Style profile reset for genre '%s'", genre)
        db.commit()


def rebuild():
    """Rebuild all style profiles from current library elements.

    Computes fresh profiles by averaging style signals across all elements
    grouped by subject_type. Unlike incremental EMA updates, this produces
    profiles that accurately reflect the current library state — essential
    after photos or elements are deleted.
    """
    from ..store import elements as element_store

    all_elements = element_store.get_all()

    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    if not all_elements:
        with lock:
            db.execute("DELETE FROM style_profiles WHERE library_id=?", (lib_id,))
            db.commit()
        log.info("Style profiles cleared (no elements in library)")
        return

    # Group elements by subject_type (genre)
    by_genre = defaultdict(list)
    for elem in all_elements:
        genre = elem.get("subject_type", "unknown") or "unknown"
        by_genre[genre].append(elem)

    today = date.today().isoformat()

    with lock:
        db.execute("DELETE FROM style_profiles WHERE library_id=?", (lib_id,))

        for genre, genre_elements in by_genre.items():
            source_photos = {e.get("source_photo") or e.get("thumbnail")
                             for e in genre_elements}
            source_photos.discard(None)
            observations = max(len(source_photos), 1)

            signals = _extract_style_signals(genre_elements)

            db.execute(
                """INSERT INTO style_profiles
                   (genre, library_id, observations, patterns, last_updated)
                   VALUES (?,?,?,?,?)""",
                (genre, lib_id, observations, json.dumps(signals), today)
            )

        db.commit()

    log.info("Style profiles rebuilt from %d elements across %d genres",
             len(all_elements), len(by_genre))


def _extract_style_signals(elements):
    """Extract style dimensions from analyzed elements.

    Groups element attributes by their relevance to style dimensions
    (lighting, composition, palette, mood, camera).

    Returns dict of {dimension: {value: weight}}.
    """
    signals = defaultdict(lambda: defaultdict(float))

    style_categories = {
        "lighting", "camera", "technique", "palette", "composition", "mood",
        "environment",
    }

    for elem in elements:
        category = elem.get("category", "")
        tags = elem.get("tags", [])

        if category in style_categories:
            for tag in tags:
                signals[category][tag] += 1.0

        attributes = elem.get("attributes", {})
        for attr_key, attr_val in attributes.items():
            if isinstance(attr_val, str):
                if attr_key in style_categories:
                    signals[attr_key][attr_val] += 0.5

    # Normalize weights within each dimension
    for dimension in signals:
        total = sum(signals[dimension].values())
        if total > 0:
            for value in signals[dimension]:
                signals[dimension][value] /= total

    return dict(signals)
