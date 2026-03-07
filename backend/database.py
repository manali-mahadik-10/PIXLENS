"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PIXLENS — database.py                                          ║
║              Pixel Intelligence & Learning ENhancement System               ║
║                                                                              ║
║  Handles ALL database operations for PIXLENS:                               ║
║    • Creating and initialising the SQLite database                          ║
║    • Managing sessions (each time you open the app = one session)           ║
║    • Saving every filter operation with its parameters & quality metrics    ║
║    • Retrieving history, statistics, and leaderboard data                   ║
║    • Bookmarking favourite results                                           ║
║    • Tagging and annotating operations                                       ║
║                                                                              ║
║  Database File: ../database/pixlens.db  (auto-created on first run)         ║
║  Tables:  sessions | filter_history | bookmarks | tags | filter_tag_map     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import os
import json
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# This builds the path to the database file no matter where you run the script from.
# __file__ = this file (database.py)
# We go one level up (..) into the database/ folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', 'database', 'pixlens.db')

# Make sure the database folder exists. If not, create it automatically.
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    """
    Open and return a connection to the SQLite database.

    row_factory = sqlite3.Row allows us to access columns by NAME like a dict:
        row['filter_name']   instead of   row[2]
    This makes the code much more readable.

    Foreign key enforcement is turned ON so that session_id in filter_history
    must always match a real id in the sessions table.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")   # enforce FK constraints
    conn.execute("PRAGMA journal_mode = WAL")   # better concurrent read/write
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE INITIALISATION  (Call once when app starts)
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """
    Create all database tables if they don't already exist.
    Safe to call every time the app starts — it won't overwrite existing data.

    TABLE OVERVIEW:
    ┌─────────────────┬──────────────────────────────────────────────────────┐
    │ sessions        │ One row per user session (app open / new project)   │
    │ filter_history  │ One row per filter operation applied                 │
    │ bookmarks       │ User-starred/favourite filter results                │
    │ tags            │ Custom labels the user can create (e.g. "sharp")    │
    │ filter_tag_map  │ Many-to-many: links filter_history rows to tags      │
    └─────────────────┴──────────────────────────────────────────────────────┘
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── TABLE 1: sessions ────────────────────────────────────────────────────
    # Tracks each working session. A session groups related filter operations.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name  TEXT    NOT NULL DEFAULT 'Untitled Session',
            description   TEXT    DEFAULT '',
            image_count   INTEGER DEFAULT 0,
            created_at    TEXT    DEFAULT (datetime('now', 'localtime')),
            updated_at    TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # ── TABLE 2: filter_history ───────────────────────────────────────────────
    # Core table. One row per filter operation applied.
    # Links back to sessions via session_id (foreign key).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filter_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER NOT NULL,
            filter_name    TEXT    NOT NULL,
            filter_category TEXT   DEFAULT 'Spatial',
            parameters     TEXT    DEFAULT '{}',
            input_image    TEXT    DEFAULT '',
            output_image   TEXT    DEFAULT '',
            width          INTEGER DEFAULT 0,
            height         INTEGER DEFAULT 0,
            psnr           REAL    DEFAULT 0.0,
            mse            REAL    DEFAULT 0.0,
            mae            REAL    DEFAULT 0.0,
            std_dev        REAL    DEFAULT 0.0,
            process_time_ms REAL   DEFAULT 0.0,
            notes          TEXT    DEFAULT '',
            applied_at     TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)

    # ── TABLE 3: bookmarks ────────────────────────────────────────────────────
    # Lets the user star/bookmark their favourite filter results to revisit.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filter_history_id INTEGER NOT NULL UNIQUE,
            label            TEXT    DEFAULT 'Favourite',
            bookmarked_at    TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (filter_history_id) REFERENCES filter_history(id) ON DELETE CASCADE
        )
    """)

    # ── TABLE 4: tags ─────────────────────────────────────────────────────────
    # Custom user-defined labels, e.g. "sharp", "blurry", "for-report", etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name   TEXT    NOT NULL UNIQUE,
            color      TEXT    DEFAULT '#00e5ff',
            created_at TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # ── TABLE 5: filter_tag_map ───────────────────────────────────────────────
    # Many-to-many: one filter operation can have many tags; one tag can apply
    # to many filter operations.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filter_tag_map (
            filter_history_id INTEGER NOT NULL,
            tag_id            INTEGER NOT NULL,
            PRIMARY KEY (filter_history_id, tag_id),
            FOREIGN KEY (filter_history_id) REFERENCES filter_history(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id)            REFERENCES tags(id)           ON DELETE CASCADE
        )
    """)

    # ── INDEXES ───────────────────────────────────────────────────────────────
    # Speed up the most common queries.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_session  ON filter_history(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_filter   ON filter_history(filter_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_date     ON filter_history(applied_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_psnr     ON filter_history(psnr)")

    conn.commit()
    conn.close()
    print(f"[PIXLENS DB] Initialised successfully → {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_session(name="New Session", description=""):
    """
    Create a new session record.
    Returns the session's auto-generated integer ID.

    Example:
        sid = create_session("Lab 3 - Edge Detection Experiments")
    """
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO sessions (session_name, description) VALUES (?, ?)",
        (name, description)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_session(session_id):
    """
    Fetch a single session record by its ID.
    Returns a dict or None if not found.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_sessions():
    """
    Fetch all sessions, newest first.
    Each row also includes the count of filter operations in that session.

    Returns a list of dicts.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            s.*,
            COUNT(h.id) AS operation_count,
            MAX(h.applied_at) AS last_operation_at,
            ROUND(AVG(h.psnr), 2) AS avg_psnr
        FROM sessions s
        LEFT JOIN filter_history h ON s.id = h.session_id
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_name(session_id, new_name):
    """
    Rename an existing session.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET session_name = ?, updated_at = datetime('now','localtime') WHERE id = ?",
        (new_name, session_id)
    )
    conn.commit()
    conn.close()


def delete_session(session_id):
    """
    Delete a session and ALL its filter_history rows (via CASCADE).
    Use with caution — this is permanent.
    """
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# FILTER HISTORY — SAVE & RETRIEVE
# ─────────────────────────────────────────────────────────────────────────────

def save_filter_operation(
        session_id,
        filter_name,
        filter_category="Spatial",
        parameters=None,
        input_image="",
        output_image="",
        width=0,
        height=0,
        psnr=0.0,
        mse=0.0,
        mae=0.0,
        std_dev=0.0,
        process_time_ms=0.0,
        notes=""
):
    """
    Save one filter operation to the database.
    This is called automatically every time you click 'Apply Filter' in PIXLENS.

    Parameters:
        session_id       (int)   — which session this belongs to
        filter_name      (str)   — e.g. 'gaussian', 'canny', 'sobel'
        filter_category  (str)   — e.g. 'Spatial', 'Frequency', 'Morphology'
        parameters       (dict)  — e.g. {'ksize': 5, 'sigma': 1.5}
        input_image      (str)   — filename of the source image
        output_image     (str)   — filename of the processed result
        width, height    (int)   — image dimensions in pixels
        psnr             (float) — Peak Signal-to-Noise Ratio
        mse              (float) — Mean Squared Error
        mae              (float) — Mean Absolute Error
        std_dev          (float) — Standard deviation of processed image
        process_time_ms  (float) — How many milliseconds processing took
        notes            (str)   — Any user note

    Returns:
        (int) — The auto-generated ID of the new row
    """
    if parameters is None:
        parameters = {}

    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO filter_history (
            session_id, filter_name, filter_category, parameters,
            input_image, output_image, width, height,
            psnr, mse, mae, std_dev, process_time_ms, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        filter_name,
        filter_category,
        json.dumps(parameters),   # store dict as JSON string
        input_image,
        output_image,
        width,
        height,
        round(float(psnr), 4),
        round(float(mse), 4),
        round(float(mae), 4),
        round(float(std_dev), 4),
        round(float(process_time_ms), 2),
        notes
    ))

    record_id = cursor.lastrowid

    # Also update the session's updated_at timestamp
    conn.execute(
        "UPDATE sessions SET updated_at = datetime('now','localtime') WHERE id = ?",
        (session_id,)
    )

    conn.commit()
    conn.close()
    return record_id


def get_history(limit=50, session_id=None, filter_name=None):
    """
    Retrieve filter history records, newest first.

    Filters (all optional):
        limit       — max number of rows to return (default: 50)
        session_id  — if given, only return records from this session
        filter_name — if given, only return records for this filter type

    Returns:
        list of dicts, each representing one filter operation.
        parameters is automatically parsed back from JSON to a dict.
    """
    conn = get_connection()

    query = """
        SELECT
            h.*,
            s.session_name,
            CASE WHEN b.id IS NOT NULL THEN 1 ELSE 0 END AS is_bookmarked
        FROM filter_history h
        LEFT JOIN sessions  s ON h.session_id = s.id
        LEFT JOIN bookmarks b ON h.id = b.filter_history_id
        WHERE 1=1
    """
    args = []

    if session_id is not None:
        query += " AND h.session_id = ?"
        args.append(session_id)

    if filter_name is not None:
        query += " AND h.filter_name = ?"
        args.append(filter_name)

    query += " ORDER BY h.applied_at DESC LIMIT ?"
    args.append(limit)

    rows = conn.execute(query, args).fetchall()
    conn.close()

    result = []
    for r in rows:
        row_dict = dict(r)
        # Parse parameters back from JSON string → Python dict
        try:
            row_dict['parameters'] = json.loads(row_dict['parameters'])
        except (json.JSONDecodeError, TypeError):
            row_dict['parameters'] = {}
        result.append(row_dict)

    return result


def get_single_operation(record_id):
    """
    Fetch one specific filter_history row by its ID.
    Returns a dict or None.
    """
    conn = get_connection()
    row = conn.execute("""
        SELECT h.*, s.session_name
        FROM filter_history h
        LEFT JOIN sessions s ON h.session_id = s.id
        WHERE h.id = ?
    """, (record_id,)).fetchone()
    conn.close()

    if row:
        d = dict(row)
        try:
            d['parameters'] = json.loads(d['parameters'])
        except Exception:
            d['parameters'] = {}
        return d
    return None


def update_notes(record_id, notes):
    """
    Add or update the notes field on a filter_history record.
    Useful for annotating what worked well during experiments.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE filter_history SET notes = ? WHERE id = ?",
        (notes, record_id)
    )
    conn.commit()
    conn.close()


def delete_operation(record_id):
    """
    Delete one specific filter_history record.
    """
    conn = get_connection()
    conn.execute("DELETE FROM filter_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def delete_all_history(session_id=None):
    """
    Delete ALL history records.
    If session_id is provided, only deletes records from that session.
    """
    conn = get_connection()
    if session_id:
        conn.execute("DELETE FROM filter_history WHERE session_id = ?", (session_id,))
    else:
        conn.execute("DELETE FROM filter_history")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# BOOKMARKS
# ─────────────────────────────────────────────────────────────────────────────

def add_bookmark(filter_history_id, label="Favourite"):
    """
    Bookmark (star) a specific filter result so you can find it easily later.
    If already bookmarked, update the label.

    Example:
        add_bookmark(42, label="Best Canny result for report")
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO bookmarks (filter_history_id, label)
        VALUES (?, ?)
        ON CONFLICT(filter_history_id) DO UPDATE SET label = excluded.label
    """, (filter_history_id, label))
    conn.commit()
    conn.close()


def remove_bookmark(filter_history_id):
    """
    Remove the bookmark from a filter result.
    """
    conn = get_connection()
    conn.execute(
        "DELETE FROM bookmarks WHERE filter_history_id = ?",
        (filter_history_id,)
    )
    conn.commit()
    conn.close()


def get_bookmarks():
    """
    Get all bookmarked filter operations with full details.
    Returns a list of dicts, newest bookmark first.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            b.id AS bookmark_id,
            b.label,
            b.bookmarked_at,
            h.*,
            s.session_name
        FROM bookmarks b
        JOIN filter_history h ON b.filter_history_id = h.id
        LEFT JOIN sessions  s ON h.session_id = s.id
        ORDER BY b.bookmarked_at DESC
    """).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['parameters'] = json.loads(d['parameters'])
        except Exception:
            d['parameters'] = {}
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAGS
# ─────────────────────────────────────────────────────────────────────────────

def create_tag(tag_name, color="#00e5ff"):
    """
    Create a new tag label.
    If a tag with the same name already exists, returns its existing ID.

    Example:
        tid = create_tag("sharp-result", color="#00e676")
    """
    conn = get_connection()
    # Try to insert; if name already exists, just fetch its ID
    cursor = conn.execute(
        "INSERT OR IGNORE INTO tags (tag_name, color) VALUES (?, ?)",
        (tag_name, color)
    )
    if cursor.lastrowid == 0:
        # Tag already existed — fetch its ID
        row = conn.execute("SELECT id FROM tags WHERE tag_name = ?", (tag_name,)).fetchone()
        tag_id = row['id']
    else:
        tag_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return tag_id


def tag_operation(filter_history_id, tag_name, color="#00e5ff"):
    """
    Apply a tag to a filter_history record.
    Creates the tag automatically if it doesn't exist yet.

    Example:
        tag_operation(17, "butterworth-demo")
    """
    tag_id = create_tag(tag_name, color)
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO filter_tag_map (filter_history_id, tag_id) VALUES (?, ?)",
        (filter_history_id, tag_id)
    )
    conn.commit()
    conn.close()


def untag_operation(filter_history_id, tag_name):
    """
    Remove a specific tag from a filter_history record.
    """
    conn = get_connection()
    conn.execute("""
        DELETE FROM filter_tag_map
        WHERE filter_history_id = ?
          AND tag_id = (SELECT id FROM tags WHERE tag_name = ?)
    """, (filter_history_id, tag_name))
    conn.commit()
    conn.close()


def get_all_tags():
    """
    Get all tags along with how many filter operations use each one.
    Returns a list of dicts.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT t.*, COUNT(m.filter_history_id) AS usage_count
        FROM tags t
        LEFT JOIN filter_tag_map m ON t.id = m.tag_id
        GROUP BY t.id
        ORDER BY usage_count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_operations_by_tag(tag_name, limit=50):
    """
    Get all filter history records that have a specific tag.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT h.*, s.session_name
        FROM filter_history h
        LEFT JOIN sessions s ON h.session_id = s.id
        JOIN filter_tag_map m ON h.id = m.filter_history_id
        JOIN tags t ON m.tag_id = t.id
        WHERE t.tag_name = ?
        ORDER BY h.applied_at DESC
        LIMIT ?
    """, (tag_name, limit)).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['parameters'] = json.loads(d['parameters'])
        except Exception:
            d['parameters'] = {}
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_statistics(session_id=None):
    """
    Return aggregated statistics about all filter operations.
    If session_id is given, statistics are limited to that session only.

    Returns a dict with:
        total_operations   — total number of filters applied
        unique_filters     — how many distinct filter types were used
        avg_psnr           — average PSNR across all operations
        best_psnr          — highest PSNR achieved (best quality result)
        worst_psnr         — lowest PSNR achieved
        avg_process_time   — average processing time in milliseconds
        most_used_filter   — which filter was applied most often
        total_sessions     — number of sessions in the database
        filters_breakdown  — dict of {filter_name: count} for all filters
    """
    conn = get_connection()

    where = "WHERE h.session_id = ?" if session_id else ""
    args  = (session_id,) if session_id else ()

    # Overall aggregated stats
    stats_row = conn.execute(f"""
        SELECT
            COUNT(*)                          AS total_operations,
            COUNT(DISTINCT filter_name)       AS unique_filters,
            ROUND(AVG(psnr), 2)               AS avg_psnr,
            ROUND(MAX(psnr), 2)               AS best_psnr,
            ROUND(MIN(psnr), 2)               AS worst_psnr,
            ROUND(AVG(process_time_ms), 2)    AS avg_process_time,
            ROUND(AVG(mse), 2)                AS avg_mse
        FROM filter_history h
        {where}
    """, args).fetchone()

    # Most used filter
    most_used_row = conn.execute(f"""
        SELECT filter_name, COUNT(*) AS cnt
        FROM filter_history h
        {where}
        GROUP BY filter_name
        ORDER BY cnt DESC
        LIMIT 1
    """, args).fetchone()

    # Per-filter breakdown
    breakdown_rows = conn.execute(f"""
        SELECT filter_name, COUNT(*) AS cnt, ROUND(AVG(psnr),2) AS avg_psnr
        FROM filter_history h
        {where}
        GROUP BY filter_name
        ORDER BY cnt DESC
    """, args).fetchall()

    # Total sessions count (always global)
    session_count = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()['c']

    conn.close()

    stats = dict(stats_row) if stats_row else {}
    stats['most_used_filter']  = most_used_row['filter_name'] if most_used_row else 'N/A'
    stats['total_sessions']    = session_count
    stats['filters_breakdown'] = [dict(r) for r in breakdown_rows]

    return stats


def get_best_results(limit=10, metric='psnr'):
    """
    Get the top filter operations ranked by a quality metric.

    metric options: 'psnr' (higher = better), 'mse' (lower = better)

    Returns a list of dicts, best results first.
    """
    order = "DESC" if metric == 'psnr' else "ASC"
    conn = get_connection()
    rows = conn.execute(f"""
        SELECT h.*, s.session_name
        FROM filter_history h
        LEFT JOIN sessions s ON h.session_id = s.id
        WHERE h.{metric} > 0
        ORDER BY h.{metric} {order}
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['parameters'] = json.loads(d['parameters'])
        except Exception:
            d['parameters'] = {}
        result.append(d)
    return result


def get_filter_usage_over_time(days=30):
    """
    Returns daily filter usage counts for the last N days.
    Useful for plotting a usage graph.

    Returns a list of dicts: [{'date': '2025-03-01', 'count': 12}, ...]
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            DATE(applied_at) AS date,
            COUNT(*)         AS count
        FROM filter_history
        WHERE applied_at >= DATE('now', ?)
        GROUP BY DATE(applied_at)
        ORDER BY date ASC
    """, (f'-{days} days',)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_history(keyword, limit=30):
    """
    Full-text search across filter_name, notes, and parameters.
    Useful for finding a specific past experiment.

    Example:
        results = search_history("sigma 2.5")
    """
    conn = get_connection()
    pattern = f"%{keyword}%"
    rows = conn.execute("""
        SELECT h.*, s.session_name
        FROM filter_history h
        LEFT JOIN sessions s ON h.session_id = s.id
        WHERE h.filter_name LIKE ?
           OR h.notes       LIKE ?
           OR h.parameters  LIKE ?
        ORDER BY h.applied_at DESC
        LIMIT ?
    """, (pattern, pattern, pattern, limit)).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['parameters'] = json.loads(d['parameters'])
        except Exception:
            d['parameters'] = {}
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_history_to_json(filepath=None, session_id=None):
    """
    Export the complete filter history to a JSON file.
    Useful for backing up your experiments or importing into Excel/Sheets.

    If filepath is None, saves to: database/pixlens_export_<timestamp>.json
    Returns the filepath where it was saved.
    """
    data = get_history(limit=10000, session_id=session_id)

    if filepath is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        db_dir    = os.path.dirname(DB_PATH)
        filepath  = os.path.join(db_dir, f"pixlens_export_{timestamp}.json")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "exported_at":   datetime.datetime.now().isoformat(),
            "total_records": len(data),
            "data":          data
        }, f, indent=2, ensure_ascii=False)

    print(f"[PIXLENS DB] Exported {len(data)} records → {filepath}")
    return filepath


def export_history_to_csv(filepath=None, session_id=None):
    """
    Export filter history to a CSV file — easy to open in Excel.
    Returns the filepath where it was saved.
    """
    import csv

    data = get_history(limit=10000, session_id=session_id)

    if filepath is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        db_dir    = os.path.dirname(DB_PATH)
        filepath  = os.path.join(db_dir, f"pixlens_export_{timestamp}.csv")

    if not data:
        print("[PIXLENS DB] No data to export.")
        return None

    fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            row['parameters'] = json.dumps(row['parameters'])  # re-serialise for CSV
            writer.writerow(row)

    print(f"[PIXLENS DB] Exported {len(data)} records → {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_db_info():
    """
    Return metadata about the database — useful for a settings/info page.

    Returns a dict with:
        db_path        — absolute path to the .db file
        db_size_kb     — file size in kilobytes
        total_records  — total rows in filter_history
        table_counts   — dict of {table_name: row_count}
    """
    conn = get_connection()

    tables = ['sessions', 'filter_history', 'bookmarks', 'tags', 'filter_tag_map']
    table_counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
        table_counts[table] = row['c']

    conn.close()

    file_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return {
        "db_path":       os.path.abspath(DB_PATH),
        "db_size_kb":    round(file_size / 1024, 2),
        "total_records": table_counts.get('filter_history', 0),
        "table_counts":  table_counts
    }


def vacuum_db():
    """
    Run VACUUM to compact and optimise the database file.
    Safe to run at any time — does not delete any data.
    Useful after deleting many records to reclaim disk space.
    """
    conn = get_connection()
    conn.execute("VACUUM")
    conn.close()
    print("[PIXLENS DB] VACUUM complete — database optimised.")


def reset_database():
    """
    ⚠ DANGER: Drops ALL tables and recreates them empty.
    All sessions, history, bookmarks, and tags are permanently deleted.
    Only call this if you want to start completely fresh.
    """
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS filter_tag_map")
    conn.execute("DROP TABLE IF EXISTS bookmarks")
    conn.execute("DROP TABLE IF EXISTS tags")
    conn.execute("DROP TABLE IF EXISTS filter_history")
    conn.execute("DROP TABLE IF EXISTS sessions")
    conn.commit()
    conn.close()
    init_db()
    print("[PIXLENS DB] ⚠ Database has been fully reset.")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (Run this file directly: python database.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PIXLENS — database.py  Quick Test")
    print("=" * 60)

    # 1. Initialise
    init_db()

    # 2. Create a session
    sid = create_session("Test Session — Gaussian & Canny")
    print(f"\n[✓] Created session ID: {sid}")

    # 3. Save two filter operations
    op1 = save_filter_operation(
        session_id=sid,
        filter_name="gaussian",
        filter_category="Spatial",
        parameters={"ksize": 5, "sigma": 1.5},
        input_image="test.png",
        output_image="output_gaussian.png",
        width=512, height=512,
        psnr=38.21, mse=9.81, mae=2.14, std_dev=45.3,
        process_time_ms=12.4,
        notes="Good smoothing result"
    )
    print(f"[✓] Saved operation 1 (Gaussian) → row ID: {op1}")

    op2 = save_filter_operation(
        session_id=sid,
        filter_name="canny",
        filter_category="Edge Detection",
        parameters={"sigma": 1.0, "lowT": 20, "highT": 60},
        input_image="test.png",
        output_image="output_canny.png",
        width=512, height=512,
        psnr=22.5, mse=365.1, mae=15.3, std_dev=88.7,
        process_time_ms=45.2,
        notes="Sharp edges detected"
    )
    print(f"[✓] Saved operation 2 (Canny) → row ID: {op2}")

    # 4. Bookmark one result
    add_bookmark(op1, label="Best Gaussian — use in report")
    print(f"[✓] Bookmarked operation {op1}")

    # 5. Tag an operation
    tag_operation(op2, "edge-detection-demo", color="#ff6b35")
    print(f"[✓] Tagged operation {op2} with 'edge-detection-demo'")

    # 6. Retrieve history
    history = get_history(limit=10)
    print(f"\n[✓] History ({len(history)} records):")
    for row in history:
        print(f"    • [{row['id']}] {row['filter_name']} | PSNR={row['psnr']} | {row['applied_at']}")

    # 7. Statistics
    stats = get_statistics(session_id=sid)
    print(f"\n[✓] Statistics for session {sid}:")
    print(f"    Total operations : {stats['total_operations']}")
    print(f"    Average PSNR     : {stats['avg_psnr']} dB")
    print(f"    Best PSNR        : {stats['best_psnr']} dB")
    print(f"    Most used filter : {stats['most_used_filter']}")

    # 8. DB Info
    info = get_db_info()
    print(f"\n[✓] Database info:")
    print(f"    Path    : {info['db_path']}")
    print(f"    Size    : {info['db_size_kb']} KB")
    print(f"    Tables  : {info['table_counts']}")

    # 9. Export
    json_path = export_history_to_json()
    csv_path  = export_history_to_csv()

    print(f"\n[✓] Exported to JSON : {json_path}")
    print(f"[✓] Exported to CSV  : {csv_path}")

    print("\n" + "=" * 60)
    print("  All tests passed! PIXLENS database is working correctly.")
    print("=" * 60)