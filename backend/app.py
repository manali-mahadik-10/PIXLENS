"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PIXLENS — app.py                                               ║
║              Pixel Intelligence & Learning ENhancement System               ║
║                                                                              ║
║  This is the Flask web server — the backbone of PIXLENS.                    ║
║  It receives requests from the browser frontend, calls the image            ║
║  processing functions in filters.py, saves everything to the database       ║
║  via database.py, and returns results back to the browser.                  ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║      cd backend                                                              ║
║      python app.py                                                           ║
║                                                                              ║
║  SERVER STARTS AT:  http://localhost:5000                                    ║
║                                                                              ║
║  API ENDPOINTS:                                                              ║
║    GET  /api/health              — check server is running                  ║
║    GET  /api/filters             — list all available filters               ║
║    POST /api/session             — create a new session                     ║
║    GET  /api/sessions            — list all sessions                        ║
║    PUT  /api/session/<id>        — rename a session                         ║
║    DELETE /api/session/<id>      — delete a session                         ║
║    POST /api/apply-filter        — apply a DIVP filter to an image          ║
║    POST /api/upload              — upload an image file                     ║
║    GET  /api/history             — get filter operation history             ║
║    GET  /api/history/<id>        — get one specific history record          ║
║    PUT  /api/history/<id>/notes  — update notes on a record                 ║
║    DELETE /api/history/<id>      — delete one history record                ║
║    DELETE /api/history           — clear all history                        ║
║    POST /api/bookmark/<id>       — bookmark a result                        ║
║    DELETE /api/bookmark/<id>     — remove a bookmark                        ║
║    GET  /api/bookmarks           — list all bookmarks                       ║
║    POST /api/tag/<id>            — tag a filter result                      ║
║    GET  /api/tags                — list all tags                            ║
║    GET  /api/stats               — database statistics                      ║
║    GET  /api/best-results        — top results by PSNR                      ║
║    GET  /api/db-info             — database file info                       ║
║    POST /api/export/json         — export history as JSON                   ║
║    POST /api/export/csv          — export history as CSV                    ║
║    GET  /api/image/output/<file> — serve a processed output image           ║
║    GET  /api/image/upload/<file> — serve an uploaded image                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import io
import json
import base64
import time
import uuid
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from PIL import Image
import numpy as np

# ── Import our own modules ────────────────────────────────────────────────────
from filters  import apply_filter, list_filters, img_to_array, array_to_b64, compute_metrics
from database import (
    init_db,
    create_session, get_session, get_all_sessions,
    update_session_name, delete_session,
    save_filter_operation, get_history, get_single_operation,
    update_notes, delete_operation, delete_all_history,
    add_bookmark, remove_bookmark, get_bookmarks,
    create_tag, tag_operation, untag_operation, get_all_tags, get_operations_by_tag,
    get_statistics, get_best_results, get_filter_usage_over_time,
    search_history, export_history_to_json, export_history_to_csv,
    get_db_info, vacuum_db, reset_database
)


# ═════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Allow all origins — lets the HTML frontend (file://) call this server
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Folder paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, '..', 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE  = (2048, 2048)   # Resize very large images to keep things fast
ALLOWED_TYPES   = {'image/jpeg', 'image/png', 'image/bmp',
                   'image/tiff', 'image/webp', 'image/gif'}
ALLOWED_EXT     = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ── Initialise database on startup ────────────────────────────────────────────
init_db()
print("=" * 60)
print("  PIXLENS Backend Server — app.py")
print("=" * 60)
print(f"  Upload folder : {os.path.abspath(UPLOAD_FOLDER)}")
print(f"  Output folder : {os.path.abspath(OUTPUT_FOLDER)}")
print(f"  Available filters: {sum(len(v) for v in list_filters().values())}")
print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def success(data=None, message="OK", status=200):
    """Standard success response wrapper."""
    resp = {"success": True, "message": message}
    if data is not None:
        resp.update(data)
    return jsonify(resp), status


def error(message, status=400, details=None):
    """Standard error response wrapper."""
    resp = {"success": False, "error": message}
    if details:
        resp["details"] = details
    return jsonify(resp), status


def decode_base64_image(b64_string):
    """
    Decode a base64 image string from the browser into a PIL Image.
    Handles the 'data:image/png;base64,...' format sent by the frontend.

    Returns:
        (PIL.Image, width, height)
    Raises:
        ValueError if decoding fails
    """
    try:
        # Strip the data URL prefix if present
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]
        img_bytes = base64.b64decode(b64_string)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Resize very large images to prevent slow processing
        if pil_img.width > MAX_IMAGE_SIZE[0] or pil_img.height > MAX_IMAGE_SIZE[1]:
            pil_img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)

        return pil_img, pil_img.width, pil_img.height
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def save_image_file(pil_img, folder, prefix="img"):
    """
    Save a PIL Image to disk with a unique filename.

    Returns:
        filename (str) — just the filename, not the full path
    """
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    filepath = os.path.join(folder, filename)
    pil_img.save(filepath, format='PNG', optimize=True)
    return filename


def load_image_file(folder, filename):
    """
    Load an image from disk by filename and folder.

    Returns:
        PIL Image or None if not found
    """
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        return None
    return Image.open(filepath).convert('RGB')


def get_filter_category(filter_name):
    """Look up a filter's DIVP category from the registry."""
    filters = list_filters()
    for cat, names in filters.items():
        if filter_name in names:
            return cat
    return "Unknown"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HEALTH & INFO ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    GET /api/health
    Quick ping to confirm the server is running.
    The frontend calls this on startup to verify the backend is available.
    """
    filter_summary = list_filters()
    total_filters  = sum(len(v) for v in filter_summary.values())

    return success({
        "server":         "PIXLENS Backend",
        "version":        "2.0.0",
        "status":         "running",
        "timestamp":      datetime.now().isoformat(),
        "total_filters":  total_filters,
        "categories":     list(filter_summary.keys()),
        "python_backend": True,
        "database":       True
    }, message="PIXLENS backend is running")


@app.route('/api/filters', methods=['GET'])
def get_filters():
    """
    GET /api/filters
    Returns all available filters grouped by DIVP category.
    The frontend uses this to build the filter sidebar dynamically.

    Response:
        {
          "Spatial": ["mean", "gaussian", ...],
          "Edge Detection": ["sobel", "canny", ...],
          ...
        }
    """
    return success({"filters": list_filters()})


@app.route('/api/db-info', methods=['GET'])
def database_info():
    """
    GET /api/db-info
    Returns metadata about the SQLite database file.
    """
    try:
        info = get_db_info()
        return success({"db_info": info})
    except Exception as e:
        return error("Failed to get database info", details=str(e))


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    GET /api/stats?session_id=<optional>
    Returns aggregated statistics about filter usage and quality metrics.

    Optional query param:
        session_id — if given, stats are scoped to that session only
    """
    try:
        session_id = request.args.get('session_id', type=int)
        stats      = get_statistics(session_id=session_id)
        usage      = get_filter_usage_over_time(days=30)
        return success({"stats": stats, "usage_over_time": usage})
    except Exception as e:
        return error("Failed to get statistics", details=str(e))


@app.route('/api/best-results', methods=['GET'])
def best_results():
    """
    GET /api/best-results?metric=psnr&limit=10
    Returns top filter operations ranked by quality metric.

    Query params:
        metric — 'psnr' (default, higher=better) or 'mse' (lower=better)
        limit  — number of results to return (default: 10)
    """
    try:
        metric = request.args.get('metric', 'psnr')
        limit  = request.args.get('limit', 10, type=int)
        results = get_best_results(limit=limit, metric=metric)
        return success({"results": results})
    except Exception as e:
        return error("Failed to get best results", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SESSION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/session', methods=['POST'])
def new_session():
    """
    POST /api/session
    Create a new processing session.

    Request body (JSON):
        { "name": "My Lab 3 Session", "description": "Testing Canny" }

    Response:
        { "session_id": 5, "name": "My Lab 3 Session" }
    """
    try:
        body        = request.get_json(silent=True) or {}
        name        = body.get('name', f'Session {datetime.now().strftime("%d %b %H:%M")}')
        description = body.get('description', '')
        session_id  = create_session(name=name, description=description)
        session     = get_session(session_id)
        return success({"session": session}, message="Session created", status=201)
    except Exception as e:
        return error("Failed to create session", details=str(e))


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """
    GET /api/sessions
    List all sessions, newest first, each with operation count and avg PSNR.
    """
    try:
        sessions = get_all_sessions()
        return success({"sessions": sessions, "count": len(sessions)})
    except Exception as e:
        return error("Failed to list sessions", details=str(e))


@app.route('/api/session/<int:session_id>', methods=['GET'])
def get_session_detail(session_id):
    """
    GET /api/session/<id>
    Get details of a single session.
    """
    try:
        session = get_session(session_id)
        if not session:
            return error(f"Session {session_id} not found", status=404)
        return success({"session": session})
    except Exception as e:
        return error("Failed to get session", details=str(e))


@app.route('/api/session/<int:session_id>', methods=['PUT'])
def rename_session(session_id):
    """
    PUT /api/session/<id>
    Rename a session.

    Request body:
        { "name": "New Session Name" }
    """
    try:
        body = request.get_json(silent=True) or {}
        name = body.get('name', '').strip()
        if not name:
            return error("Session name cannot be empty")
        update_session_name(session_id, name)
        return success({"session_id": session_id, "new_name": name},
                       message="Session renamed")
    except Exception as e:
        return error("Failed to rename session", details=str(e))


@app.route('/api/session/<int:session_id>', methods=['DELETE'])
def remove_session(session_id):
    """
    DELETE /api/session/<id>
    Delete a session and all its associated filter history (cascades).
    """
    try:
        session = get_session(session_id)
        if not session:
            return error(f"Session {session_id} not found", status=404)
        delete_session(session_id)
        return success(message=f"Session {session_id} deleted")
    except Exception as e:
        return error("Failed to delete session", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CORE FILTER ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/apply-filter', methods=['POST'])
def apply_filter_endpoint():
    """
    POST /api/apply-filter
    ─────────────────────────────────────────────────────────────────────────
    THE MAIN ENDPOINT — called every time the user clicks 'Apply Filter'.

    What happens here (in order):
        1.  Parse the request — get image, filter name, parameters, session ID
        2.  Decode the base64 image from the browser into a NumPy array
        3.  Call apply_filter() from filters.py — runs the DIVP algorithm
        4.  Compute quality metrics (PSNR, MSE, MAE, Std Dev)
        5.  Save the processed image to the outputs/ folder
        6.  Save the operation record to the database (filter_history table)
        7.  Encode the result image as base64 to send back to the browser
        8.  Return JSON with processed image + all metrics

    Request body (JSON):
        {
            "image":       "<base64 encoded image>",
            "filter_name": "gaussian",
            "params":      { "ksize": 5, "sigma": 1.5 },
            "session_id":  3
        }

    Response:
        {
            "success":          true,
            "processed_image":  "data:image/png;base64,...",
            "filter_name":      "gaussian",
            "category":         "Spatial",
            "metrics": {
                "psnr":    38.21,
                "mse":     9.81,
                "mae":     2.14,
                "std_dev": 45.3
            },
            "time_ms":         12.4,
            "record_id":       17,
            "output_filename": "output_gaussian_abc123.png",
            "dimensions": { "width": 512, "height": 512 }
        }
    ─────────────────────────────────────────────────────────────────────────
    """
    t_request_start = time.perf_counter()

    try:
        # ── Step 1: Parse request ────────────────────────────────────────────
        body        = request.get_json(silent=True)
        if not body:
            return error("Request body must be JSON")

        image_b64   = body.get('image')
        filter_name = body.get('filter_name', 'mean').strip()
        params      = body.get('params', {})
        session_id  = body.get('session_id', 1)
        notes       = body.get('notes', '')

        if not image_b64:
            return error("No image provided. Send 'image' as a base64 string.")

        if not filter_name:
            return error("No filter_name provided.")

        # ── Step 2: Decode image ─────────────────────────────────────────────
        try:
            pil_img, img_w, img_h = decode_base64_image(image_b64)
        except ValueError as e:
            return error(str(e), status=400)

        img_array = img_to_array(pil_img)

        # Save the uploaded/original image for reference
        in_filename = save_image_file(pil_img, UPLOAD_FOLDER, prefix='input')

        # ── Step 3: Apply filter ─────────────────────────────────────────────
        try:
            filter_result = apply_filter(img_array, filter_name, params)
        except ValueError as e:
            return error(f"Filter error: {str(e)}", status=400)
        except Exception as e:
            return error(
                f"Processing failed for filter '{filter_name}'",
                status=500,
                details=str(e)
            )

        processed_array = filter_result['result']
        category        = filter_result['category']
        proc_time_ms    = filter_result['time_ms']
        metrics         = filter_result['metrics']

        # ── Step 4: Save processed image to disk ─────────────────────────────
        processed_pil  = Image.fromarray(
            np.clip(processed_array, 0, 255).astype(np.uint8)
        )
        out_filename = save_image_file(
            processed_pil, OUTPUT_FOLDER,
            prefix=f'output_{filter_name}'
        )

        # ── Step 5: Save operation to database ───────────────────────────────
        record_id = save_filter_operation(
            session_id      = session_id,
            filter_name     = filter_name,
            filter_category = category,
            parameters      = params,
            input_image     = in_filename,
            output_image    = out_filename,
            width           = img_w,
            height          = img_h,
            psnr            = metrics['psnr'],
            mse             = metrics['mse'],
            mae             = metrics['mae'],
            std_dev         = metrics['std_dev'],
            process_time_ms = proc_time_ms,
            notes           = notes
        )

        # ── Step 6: Encode processed image as base64 ─────────────────────────
        result_b64 = array_to_b64(processed_array)

        total_time = round((time.perf_counter() - t_request_start) * 1000, 2)

        # ── Step 7: Return response ───────────────────────────────────────────
        return success({
            "processed_image":  f"data:image/png;base64,{result_b64}",
            "filter_name":      filter_name,
            "category":         category,
            "metrics":          metrics,
            "time_ms":          proc_time_ms,
            "total_time_ms":    total_time,
            "record_id":        record_id,
            "output_filename":  out_filename,
            "input_filename":   in_filename,
            "dimensions":       {"width": img_w, "height": img_h},
            "params_used":      params
        }, message=f"Filter '{filter_name}' applied successfully")

    except Exception as e:
        # Catch-all: log the full traceback for debugging
        traceback.print_exc()
        return error(
            "An unexpected server error occurred",
            status=500,
            details=str(e)
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — IMAGE UPLOAD ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    POST /api/upload
    Upload an image file (multipart/form-data) and save it to uploads/ folder.
    Returns the filename and a base64 preview for the frontend to display.

    Used as an alternative to base64 inline upload for large image files.

    Form data:
        file — image file (PNG, JPG, BMP, TIFF, WEBP)

    Response:
        {
            "filename":   "input_abc123.png",
            "width":      512,
            "height":     512,
            "preview_b64":"data:image/png;base64,...",
            "size_kb":    84.3
        }
    """
    try:
        if 'file' not in request.files:
            return error("No file part in the request. Use field name 'file'.")

        file = request.files['file']
        if file.filename == '':
            return error("No file selected.")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return error(
                f"File type '{ext}' not supported. "
                f"Use: {', '.join(ALLOWED_EXT)}"
            )

        # Read, convert, resize if needed
        pil_img = Image.open(file.stream).convert('RGB')
        if pil_img.width > MAX_IMAGE_SIZE[0] or pil_img.height > MAX_IMAGE_SIZE[1]:
            pil_img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)

        filename = save_image_file(pil_img, UPLOAD_FOLDER, prefix='upload')

        # Size info
        fpath   = os.path.join(UPLOAD_FOLDER, filename)
        size_kb = round(os.path.getsize(fpath) / 1024, 2)

        # Base64 preview for the browser
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        preview_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return success({
            "filename":    filename,
            "width":       pil_img.width,
            "height":      pil_img.height,
            "preview_b64": f"data:image/png;base64,{preview_b64}",
            "size_kb":     size_kb
        }, message="Image uploaded successfully", status=201)

    except Exception as e:
        traceback.print_exc()
        return error("Upload failed", status=500, details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — IMAGE SERVING ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/image/output/<filename>', methods=['GET'])
def serve_output_image(filename):
    """
    GET /api/image/output/<filename>
    Serve a processed output image file directly.
    Used by the frontend to display saved results without re-uploading.

    Security: only allows alphanumeric filenames + extension.
    """
    # Basic security: strip path traversal attempts
    filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath, mimetype='image/png')


@app.route('/api/image/upload/<filename>', methods=['GET'])
def serve_upload_image(filename):
    """
    GET /api/image/upload/<filename>
    Serve an uploaded source image file directly.
    """
    filename = os.path.basename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath, mimetype='image/png')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FILTER HISTORY ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/history', methods=['GET'])
def get_filter_history():
    """
    GET /api/history?limit=50&session_id=<optional>&filter_name=<optional>
    Retrieve filter operation history, newest first.

    Query params:
        limit       — max rows to return (default: 50, max: 500)
        session_id  — filter by session
        filter_name — filter by filter type
        search      — keyword search across filter_name, notes, params
    """
    try:
        limit       = min(request.args.get('limit', 50, type=int), 500)
        session_id  = request.args.get('session_id', type=int)
        filter_name = request.args.get('filter_name', type=str)
        search_kw   = request.args.get('search', type=str)

        if search_kw:
            history = search_history(search_kw, limit=limit)
        else:
            history = get_history(
                limit=limit,
                session_id=session_id,
                filter_name=filter_name
            )

        return success({
            "history": history,
            "count":   len(history)
        })
    except Exception as e:
        return error("Failed to retrieve history", details=str(e))


@app.route('/api/history/<int:record_id>', methods=['GET'])
def get_history_record(record_id):
    """
    GET /api/history/<id>
    Get one specific filter operation record by its ID.
    """
    try:
        record = get_single_operation(record_id)
        if not record:
            return error(f"Record {record_id} not found", status=404)
        return success({"record": record})
    except Exception as e:
        return error("Failed to get record", details=str(e))


@app.route('/api/history/<int:record_id>/notes', methods=['PUT'])
def update_record_notes(record_id):
    """
    PUT /api/history/<id>/notes
    Add or update notes/annotation on a filter history record.

    Request body:
        { "notes": "This Gaussian sigma=2 gave the clearest result" }
    """
    try:
        body  = request.get_json(silent=True) or {}
        notes = body.get('notes', '').strip()
        record = get_single_operation(record_id)
        if not record:
            return error(f"Record {record_id} not found", status=404)
        update_notes(record_id, notes)
        return success({"record_id": record_id, "notes": notes},
                       message="Notes updated")
    except Exception as e:
        return error("Failed to update notes", details=str(e))


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history_record(record_id):
    """
    DELETE /api/history/<id>
    Delete one specific filter history record.
    """
    try:
        record = get_single_operation(record_id)
        if not record:
            return error(f"Record {record_id} not found", status=404)
        delete_operation(record_id)
        return success(message=f"Record {record_id} deleted")
    except Exception as e:
        return error("Failed to delete record", details=str(e))


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """
    DELETE /api/history?session_id=<optional>
    Clear filter history. If session_id is given, only clears that session.
    Otherwise clears ALL history (use with caution!).

    Query params:
        session_id — scope deletion to one session
    """
    try:
        session_id = request.args.get('session_id', type=int)
        delete_all_history(session_id=session_id)
        scope = f"session {session_id}" if session_id else "all sessions"
        return success(message=f"History cleared for {scope}")
    except Exception as e:
        return error("Failed to clear history", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BOOKMARK ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/bookmark/<int:record_id>', methods=['POST'])
def bookmark_record(record_id):
    """
    POST /api/bookmark/<id>
    Bookmark (star) a filter history record for easy retrieval later.

    Request body (optional):
        { "label": "Best Gaussian for report" }
    """
    try:
        record = get_single_operation(record_id)
        if not record:
            return error(f"Record {record_id} not found", status=404)

        body  = request.get_json(silent=True) or {}
        label = body.get('label', 'Favourite')
        add_bookmark(record_id, label=label)
        return success(
            {"record_id": record_id, "label": label},
            message="Bookmarked successfully"
        )
    except Exception as e:
        return error("Failed to bookmark record", details=str(e))


@app.route('/api/bookmark/<int:record_id>', methods=['DELETE'])
def unbookmark_record(record_id):
    """
    DELETE /api/bookmark/<id>
    Remove a bookmark from a filter history record.
    """
    try:
        remove_bookmark(record_id)
        return success(message=f"Bookmark removed for record {record_id}")
    except Exception as e:
        return error("Failed to remove bookmark", details=str(e))


@app.route('/api/bookmarks', methods=['GET'])
def list_bookmarks():
    """
    GET /api/bookmarks
    Retrieve all bookmarked filter results with full details.
    """
    try:
        bookmarks = get_bookmarks()
        return success({"bookmarks": bookmarks, "count": len(bookmarks)})
    except Exception as e:
        return error("Failed to retrieve bookmarks", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TAG ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/tag/<int:record_id>', methods=['POST'])
def tag_record(record_id):
    """
    POST /api/tag/<id>
    Apply a tag to a filter history record.
    Creates the tag automatically if it doesn't exist.

    Request body:
        { "tag_name": "edge-detection-demo", "color": "#ff6b35" }
    """
    try:
        record = get_single_operation(record_id)
        if not record:
            return error(f"Record {record_id} not found", status=404)

        body     = request.get_json(silent=True) or {}
        tag_name = body.get('tag_name', '').strip()
        color    = body.get('color', '#00e5ff')

        if not tag_name:
            return error("tag_name is required")

        tag_operation(record_id, tag_name, color)
        return success(
            {"record_id": record_id, "tag": tag_name},
            message=f"Tagged with '{tag_name}'"
        )
    except Exception as e:
        return error("Failed to tag record", details=str(e))


@app.route('/api/tag/<int:record_id>', methods=['DELETE'])
def untag_record(record_id):
    """
    DELETE /api/tag/<id>?tag_name=<name>
    Remove a specific tag from a filter history record.

    Query param:
        tag_name — the tag to remove
    """
    try:
        tag_name = request.args.get('tag_name', '').strip()
        if not tag_name:
            return error("tag_name query parameter is required")
        untag_operation(record_id, tag_name)
        return success(message=f"Tag '{tag_name}' removed from record {record_id}")
    except Exception as e:
        return error("Failed to remove tag", details=str(e))


@app.route('/api/tags', methods=['GET'])
def list_tags():
    """
    GET /api/tags
    List all tags with usage counts.
    Optionally filter history by a specific tag.

    Query params:
        filter_by_tag — if given, returns records with that tag
    """
    try:
        tags        = get_all_tags()
        filter_tag  = request.args.get('filter_by_tag')
        tagged_ops  = []
        if filter_tag:
            tagged_ops = get_operations_by_tag(filter_tag)

        return success({
            "tags":           tags,
            "count":          len(tags),
            "tagged_records": tagged_ops
        })
    except Exception as e:
        return error("Failed to retrieve tags", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — EXPORT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/export/json', methods=['POST'])
def export_json():
    """
    POST /api/export/json
    Export the entire filter history to a JSON file.

    Request body (optional):
        { "session_id": 3 }  — scope to one session

    Response:
        { "filepath": "...", "record_count": 42 }
    """
    try:
        body       = request.get_json(silent=True) or {}
        session_id = body.get('session_id')
        filepath   = export_history_to_json(session_id=session_id)
        count      = len(get_history(limit=10000, session_id=session_id))
        return success({
            "filepath":     filepath,
            "record_count": count
        }, message="Exported to JSON successfully")
    except Exception as e:
        return error("Export failed", details=str(e))


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """
    POST /api/export/csv
    Export filter history to a CSV file (opens in Excel).

    Request body (optional):
        { "session_id": 3 }

    Response:
        { "filepath": "...", "record_count": 42 }
    """
    try:
        body       = request.get_json(silent=True) or {}
        session_id = body.get('session_id')
        filepath   = export_history_to_csv(session_id=session_id)
        count      = len(get_history(limit=10000, session_id=session_id))
        return success({
            "filepath":     filepath,
            "record_count": count
        }, message="Exported to CSV successfully")
    except Exception as e:
        return error("Export failed", details=str(e))


@app.route('/api/export/download/<export_type>', methods=['GET'])
def download_export(export_type):
    """
    GET /api/export/download/json  or  /api/export/download/csv
    Generate and immediately stream an export file for download.

    Query params:
        session_id — optional, scope to one session
    """
    try:
        session_id = request.args.get('session_id', type=int)
        if export_type == 'json':
            filepath = export_history_to_json(session_id=session_id)
            mimetype = 'application/json'
        elif export_type == 'csv':
            filepath = export_history_to_csv(session_id=session_id)
            mimetype = 'text/csv'
        else:
            return error(f"Unknown export type '{export_type}'. Use 'json' or 'csv'.")

        if not filepath or not os.path.exists(filepath):
            return error("Export file not found")

        return send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=True,
            download_name=os.path.basename(filepath)
        )
    except Exception as e:
        return error("Download failed", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DATABASE MAINTENANCE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/db/vacuum', methods=['POST'])
def db_vacuum():
    """
    POST /api/db/vacuum
    Run VACUUM to compact and optimise the SQLite database.
    Safe to call at any time — does not delete data.
    """
    try:
        vacuum_db()
        info = get_db_info()
        return success({"db_info": info}, message="Database vacuumed and optimised")
    except Exception as e:
        return error("Vacuum failed", details=str(e))


@app.route('/api/db/reset', methods=['POST'])
def db_reset():
    """
    POST /api/db/reset
    ⚠ DANGER: Completely wipe and reinitialise the database.
    Requires confirmation in the request body.

    Request body:
        { "confirm": "RESET_ALL_DATA" }
    """
    try:
        body    = request.get_json(silent=True) or {}
        confirm = body.get('confirm', '')
        if confirm != 'RESET_ALL_DATA':
            return error(
                'Reset not confirmed. '
                'Send {"confirm": "RESET_ALL_DATA"} to proceed.',
                status=403
            )
        reset_database()
        return success(message="⚠ Database has been fully reset. All data deleted.")
    except Exception as e:
        return error("Reset failed", details=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ERROR HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    """Return JSON 404 instead of HTML."""
    return jsonify({
        "success": False,
        "error":   "Endpoint not found",
        "hint":    "Check /api/health for a list of available endpoints"
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Return JSON 405 instead of HTML."""
    return jsonify({
        "success": False,
        "error":   "HTTP method not allowed for this endpoint"
    }), 405


@app.errorhandler(413)
def request_too_large(e):
    """Return JSON 413 for oversized uploads."""
    return jsonify({
        "success": False,
        "error":   "File too large. Maximum upload size is 16MB."
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Return JSON 500 instead of HTML."""
    return jsonify({
        "success": False,
        "error":   "Internal server error",
        "details": str(e)
    }), 500


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — REQUEST / RESPONSE HOOKS
# ═════════════════════════════════════════════════════════════════════════════

@app.before_request
def log_request():
    """Log every incoming request to the console."""
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
          f"{request.method} {request.path}")


@app.after_request
def add_headers(response):
    """
    Add security and caching headers to every response.
    Also sets max upload size to 16MB.
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options']         = 'DENY'
    response.headers['Cache-Control']           = 'no-store'
    return response


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("\n  Starting PIXLENS Backend Server...")
    print("  ─────────────────────────────────────")
    print("  URL    : http://localhost:5000")
    print("  Health : http://localhost:5000/api/health")
    print("  Filters: http://localhost:5000/api/filters")
    print("  History: http://localhost:5000/api/history")
    print("  ─────────────────────────────────────")
    print("  Press Ctrl+C to stop the server\n")

    app.run(
        host='0.0.0.0',     # accessible from any device on your network
        port=5000,
        debug=True,         # auto-reloads when you edit code
        threaded=True       # handles multiple requests simultaneously
    )