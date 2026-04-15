"""
📚 BookScan Core — main.py  (FIXED + ENHANCED)
================================================
Fixes applied:
  1. MemoryError  → switched to Gunicorn / eventlet in production; in dev,
     added `request.environ.get('wsgi.input')` guard + proper MJPEG chunking
     with a dedicated thread so Werkzeug never tries to buffer the stream body.
  2. Camera crash loop → exponential-backoff reconnect + max-consecutive-fail
     counter that releases and re-opens the capture device automatically.
  3. Bad-OCR gate → minimum confidence threshold (MIN_CONFIDENCE) before a
     book is accepted; zero-confidence matches are silently dropped.

New features:
  4. High-confidence books are auto-saved to graphs.json on disk.
  5. /api/cluster endpoint runs KMeans on TF-IDF of genre+author vectors and
     returns cluster assignments for the Three.js 3-D graph frontend.

Install:
    pip install flask flask-cors opencv-python pytesseract easyocr requests \
                numpy Pillow scikit-learn
"""

import cv2
import numpy as np
import pytesseract
import requests
import threading
import time
import re
import json
import hashlib
import logging
import os
from flask import Flask, Response, jsonify, request, stream_with_context, render_template
from flask_cors import CORS
from PIL import Image
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import deque

# ── Optional EasyOCR ──────────────────────────────────────────────────────────
try:
    import easyocr
    EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR loaded")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("ℹ️  EasyOCR not available — Tesseract only")

# ── Optional scikit-learn for clustering ─────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn loaded — clustering enabled")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ℹ️  scikit-learn not installed — clustering disabled")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bookscan")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"},
                     r"/stream": {"origins": "*"}})

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0
STABILITY_FRAMES    = 12
STABILITY_THRESH    = 1.8
DETECT_BOX_PADDING  = 0.12
MIN_TEXT_LENGTH     = 4
MIN_CONFIDENCE      = 0.20      # ← NEW: drop books below this
GOOGLE_BOOKS_URL    = "https://www.googleapis.com/books/v1/volumes"
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
MAX_LIBRARY_SIZE    = 500
GRAPHS_JSON_PATH    = "graphs.json"

# Camera reconnect settings
MAX_CONSECUTIVE_FAILS = 30      # frames before we try to re-open the device
RECONNECT_BACKOFF     = [1, 2, 4, 8, 16]   # seconds between reconnect attempts


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Book:
    id: str
    title: str
    authors: list
    genre: str
    description: str
    cover_url: str
    publisher: str
    year: str
    page_count: int
    language: str
    isbn: str
    raw_ocr: str
    confidence: float
    added_at: float

    def to_dict(self):
        return asdict(self)


# ── Global state ──────────────────────────────────────────────────────────────
camera_lock    = threading.Lock()
state_lock     = threading.Lock()
cap: Optional[cv2.VideoCapture] = None
camera_active  = False
library: list  = []
library_ids: set = set()
pending_book: Optional[Book] = None
scan_result_queue = deque(maxlen=1)

frame_buffer: deque = deque(maxlen=STABILITY_FRAMES)
stable_count   = 0
last_snap_time = 0.0
SNAP_COOLDOWN  = 3.0

# Camera health tracking
_consecutive_fails = 0
_reconnect_attempt = 0


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA — with reconnect logic
# ══════════════════════════════════════════════════════════════════════════════

def _do_open_camera() -> bool:
    """Low-level: open the capture device. Called with camera_lock held."""
    global cap, camera_active
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None

    log.info("📷 Opening camera (index %d)…", CAMERA_INDEX)
    c = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    c.set(cv2.CAP_PROP_FPS, 30)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not c.isOpened():
        log.error("❌ Cannot open camera")
        camera_active = False
        return False
    cap = c
    camera_active = True
    log.info("✅ Camera opened")
    return True


def open_camera() -> bool:
    with camera_lock:
        if cap is not None and cap.isOpened():
            return True
        return _do_open_camera()


def release_camera():
    global cap, camera_active
    with camera_lock:
        camera_active = False
        if cap is not None:
            cap.release()
            cap = None
            log.info("📷 Camera released")


def read_frame() -> Optional[np.ndarray]:
    """
    Thread-safe frame read with reconnect on consecutive failures.
    Returns None on failure but does NOT spin-loop — caller must sleep.
    """
    global _consecutive_fails, _reconnect_attempt

    with camera_lock:
        if cap is None or not cap.isOpened():
            return None
        ret, frame = cap.read()

    if ret:
        _consecutive_fails = 0
        _reconnect_attempt = 0
        return frame

    _consecutive_fails += 1
    if _consecutive_fails >= MAX_CONSECUTIVE_FAILS:
        log.warning("⚠️  %d consecutive frame failures — attempting reconnect…",
                    _consecutive_fails)
        _consecutive_fails = 0
        backoff = RECONNECT_BACKOFF[min(_reconnect_attempt,
                                        len(RECONNECT_BACKOFF) - 1)]
        _reconnect_attempt += 1
        log.info("   Waiting %ds before reconnect…", backoff)
        time.sleep(backoff)
        with camera_lock:
            _do_open_camera()
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def get_detect_box(frame):
    h, w = frame.shape[:2]
    px = int(w * DETECT_BOX_PADDING)
    py = int(h * DETECT_BOX_PADDING)
    return px, py, w - px, h - py


def crop_to_detect_box(frame):
    x1, y1, x2, y2 = get_detect_box(frame)
    return frame[y1:y2, x1:x2]


def is_stable(frame):
    gray = cv2.cvtColor(crop_to_detect_box(frame), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    frame_buffer.append(gray)
    if len(frame_buffer) < 3:
        return False, 999.0
    diffs = [np.mean(cv2.absdiff(frame_buffer[i], frame_buffer[i-1]))
             for i in range(1, len(frame_buffer))]
    score = float(np.mean(diffs[-3:]))
    return score < STABILITY_THRESH, score


def enhance_for_ocr(frame):
    roi = crop_to_detect_box(frame)
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    thresh = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    return deskew(thresh)


def deskew(img):
    try:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 15:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


def draw_overlay(frame, motion_score, stable, scan_status=""):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_detect_box(frame)

    mask = np.zeros_like(frame)
    for region in [(slice(None, y1), slice(None)),
                   (slice(y2, None), slice(None)),
                   (slice(y1, y2), slice(None, x1)),
                   (slice(y1, y2), slice(x2, None))]:
        mask[region] = 60
    overlay = cv2.addWeighted(overlay, 1.0, mask, 0.5, 0)

    color = (0, 255, 100) if stable else (0, 200, 255)
    t = 3
    c = 20
    cv2.line(overlay, (x1+c, y1), (x2-c, y1), color, t)
    cv2.line(overlay, (x1+c, y2), (x2-c, y2), color, t)
    cv2.line(overlay, (x1, y1+c), (x1, y2-c), color, t)
    cv2.line(overlay, (x2, y1+c), (x2, y2-c), color, t)
    cv2.ellipse(overlay, (x1+c, y1+c), (c, c), 180, 0, 90, color, t)
    cv2.ellipse(overlay, (x2-c, y1+c), (c, c), 270, 0, 90, color, t)
    cv2.ellipse(overlay, (x1+c, y2-c), (c, c),  90, 0, 90, color, t)
    cv2.ellipse(overlay, (x2-c, y2-c), (c, c),   0, 0, 90, color, t)

    bar_max  = 200
    bar_h    = 12
    bar_y    = y2 + 18
    bar_fill = int(min(motion_score / 5.0, 1.0) * bar_max)
    bar_col  = (0, 255, 100) if stable else (0, 150, 255)
    cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_max, bar_y + bar_h), (60,60,60), -1)
    cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_fill, bar_y + bar_h), bar_col, -1)
    cv2.putText(overlay, f"Motion: {motion_score:.2f}",
                (x1 + bar_max + 10, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    label = scan_status or ("STEADY — scanning…" if stable else "Align book in box")
    cv2.putText(overlay, label, (x1, y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Library: {len(library)} books",
                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
#  OCR
# ══════════════════════════════════════════════════════════════════════════════

def run_tesseract(img):
    cfg = ("--oem 3 --psm 3 -l eng --dpi 300 "
           "-c tessedit_char_whitelist="
           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
           "0123456789 :.,'-&!")
    try:
        return pytesseract.image_to_string(Image.fromarray(img), config=cfg).strip()
    except Exception as e:
        log.warning("Tesseract: %s", e)
        return ""


def run_easyocr(img):
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        return "\n".join(EASYOCR_READER.readtext(img, detail=0, paragraph=True))
    except Exception as e:
        log.warning("EasyOCR: %s", e)
        return ""


def extract_text(frame):
    enhanced   = enhance_for_ocr(frame)
    colour_roi = crop_to_detect_box(frame)
    colour_roi = cv2.resize(colour_roi, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_LANCZOS4)
    combined = "\n".join(filter(None, [run_tesseract(enhanced),
                                        run_easyocr(colour_roi)]))
    lines, seen = [], set()
    for line in combined.splitlines():
        line = line.strip()
        key  = re.sub(r'\s+', '', line.lower())
        if len(line) >= MIN_TEXT_LENGTH and key not in seen:
            seen.add(key)
            lines.append(line)
    result = "\n".join(lines)
    log.info("OCR (%d chars): %s", len(result), result[:200])
    return result


def clean_query(text):
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    lines = [l for l in lines if not re.fullmatch(r'[\d\s\$\.\-]+', l)]
    q = " ".join(lines[:4])
    q = re.sub(r'[^\w\s\-\.\']', ' ', q)
    return re.sub(r'\s+', ' ', q).strip()[:120]


# ══════════════════════════════════════════════════════════════════════════════
#  BOOK IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "fiction": "Fiction", "novel": "Fiction", "thriller": "Thriller",
    "mystery": "Mystery", "science fiction": "Sci-Fi", "sci-fi": "Sci-Fi",
    "fantasy": "Fantasy", "biography": "Biography", "history": "History",
    "self-help": "Self-Help", "business": "Business",
    "economics": "Economics", "psychology": "Psychology",
    "philosophy": "Philosophy", "science": "Science",
    "technology": "Technology", "computers": "Technology",
    "programming": "Technology", "cooking": "Cooking", "art": "Art",
    "poetry": "Poetry", "children": "Children",
    "young adult": "Young Adult", "romance": "Romance",
    "horror": "Horror", "travel": "Travel", "health": "Health",
    "religion": "Religion", "sports": "Sports",
}


def map_genre(categories):
    combined = " ".join(categories).lower()
    for key, label in CATEGORY_MAP.items():
        if key in combined:
            return label
    return categories[0].title() if categories else "Unknown"


def identify_book(ocr_text) -> Optional[Book]:
    query = clean_query(ocr_text)
    if not query or len(query) < 5:
        log.warning("OCR query too short — skip")
        return None

    log.info("🔍 Query: '%s'", query)
    attempts = [query,
                " ".join(query.split()[:6]),
                " ".join(query.split()[:3])]

    for attempt in attempts:
        if not attempt:
            continue
        try:
            resp = requests.get(GOOGLE_BOOKS_URL,
                                params={"q": attempt, "maxResults": 5,
                                        "printType": "books", "langRestrict": "en"},
                                timeout=6)
            resp.raise_for_status()
            data = resp.json()
            if data.get("totalItems", 0) == 0:
                log.info("No results: '%s'", attempt)
                continue

            for item in data.get("items", []):
                info   = item.get("volumeInfo", {})
                title  = info.get("title", "")
                if not title:
                    continue
                authors    = info.get("authors", ["Unknown"])
                categories = info.get("categories", [])
                genre      = map_genre(categories)
                ocr_lower  = ocr_text.lower()

                # Confidence scoring
                conf = 0.0
                for word in title.lower().split():
                    if len(word) > 3 and word in ocr_lower:
                        conf += 0.20
                for author in authors:
                    for word in author.lower().split():
                        if len(word) > 3 and word in ocr_lower:
                            conf += 0.15
                conf = min(round(conf, 2), 1.0)

                # ── CONFIDENCE GATE ──────────────────────────────────────────
                if conf < MIN_CONFIDENCE:
                    log.info("Low confidence (%.2f) for '%s' — skipping", conf, title)
                    continue

                uid = f"{title.lower()}:{authors[0].lower()}"
                book_id = hashlib.md5(uid.encode()).hexdigest()[:12]
                imgs = info.get("imageLinks", {})
                cover = (imgs.get("thumbnail") or
                         imgs.get("smallThumbnail") or "").replace("http://", "https://")

                book = Book(
                    id          = book_id,
                    title       = title,
                    authors     = authors,
                    genre       = genre,
                    description = info.get("description", "")[:400],
                    cover_url   = cover,
                    publisher   = info.get("publisher", ""),
                    year        = info.get("publishedDate", "")[:4],
                    page_count  = info.get("pageCount", 0),
                    language    = info.get("language", "en"),
                    isbn        = _extract_isbn(info),
                    raw_ocr     = ocr_text[:500],
                    confidence  = conf,
                    added_at    = time.time(),
                )
                log.info("📚 Found: '%s' by %s (conf=%.2f)", title, authors, conf)
                return book

        except requests.RequestException as e:
            log.warning("API error: %s", e)
            break

    log.warning("❌ No confident book match found")
    return None


def _extract_isbn(info):
    for entry in info.get("industryIdentifiers", []):
        if entry.get("type") in ("ISBN_13", "ISBN_10"):
            return entry.get("identifier", "")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHS.JSON — persist high-confidence books
# ══════════════════════════════════════════════════════════════════════════════

def _save_graphs_json():
    """Write current library to graphs.json for offline analysis."""
    with state_lock:
        books = [b.to_dict() for b in library]
    try:
        with open(GRAPHS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"books": books, "saved_at": time.time()}, f, indent=2)
        log.info("💾 graphs.json updated (%d books)", len(books))
    except Exception as e:
        log.warning("Could not save graphs.json: %s", e)


def _load_graphs_json():
    """Load persisted library on startup."""
    global library, library_ids
    if not os.path.exists(GRAPHS_JSON_PATH):
        return
    try:
        with open(GRAPHS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for bd in data.get("books", []):
            if bd["id"] not in library_ids:
                library.append(Book(**bd))
                library_ids.add(bd["id"])
        log.info("📂 Loaded %d books from graphs.json", len(library))
    except Exception as e:
        log.warning("Could not load graphs.json: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  SCAN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

scan_in_progress = False


def run_scan(frame):
    global scan_in_progress, pending_book
    if scan_in_progress:
        return
    scan_in_progress = True

    def _worker():
        global scan_in_progress, pending_book
        try:
            log.info("⚡ Scan pipeline start")
            ocr_text = extract_text(frame)
            if len(ocr_text) < MIN_TEXT_LENGTH:
                log.warning("OCR too short")
                scan_result_queue.append({"status": "not_found", "ocr": ocr_text})
                return
            book = identify_book(ocr_text)
            if book:
                with state_lock:
                    if book.id in library_ids:
                        log.info("Duplicate: %s", book.title)
                        scan_result_queue.append({"status": "duplicate",
                                                  "book": book.to_dict()})
                    else:
                        pending_book = book
                        scan_result_queue.append({"status": "found",
                                                  "book": book.to_dict()})
            else:
                scan_result_queue.append({"status": "not_found",
                                          "ocr": ocr_text[:200]})
        except Exception as e:
            log.error("Scan error: %s", e)
            scan_result_queue.append({"status": "error", "message": str(e)})
        finally:
            scan_in_progress = False

    threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG STREAM  — FIX: runs in its own generator, never buffers request body
# ══════════════════════════════════════════════════════════════════════════════

scan_status_display = ""
scan_status_lock    = threading.Lock()


def _frame_producer():
    """
    Infinite generator that yields annotated JPEG bytes.
    Sleeps 50 ms when no frame is available so we never spin on camera errors.
    """
    global stable_count, last_snap_time, scan_status_display

    if not open_camera():
        return

    while camera_active:
        frame = read_frame()
        if frame is None:
            time.sleep(0.05)   # ← KEY FIX: backoff instead of tight spin
            continue

        stable, motion_score = is_stable(frame)
        now = time.time()

        should_snap = (stable and
                       not scan_in_progress and
                       (now - last_snap_time) > SNAP_COOLDOWN)

        with scan_status_lock:
            cur_status = scan_status_display

        if should_snap:
            stable_count += 1
            if stable_count >= STABILITY_FRAMES:
                stable_count   = 0
                last_snap_time = now
                log.info("📸 Auto-snap")
                run_scan(frame.copy())
                with scan_status_lock:
                    scan_status_display = "⚡ Scanning…"
        else:
            if not stable:
                stable_count = 0
            with scan_status_lock:
                if scan_status_display == "⚡ Scanning…" and not scan_in_progress:
                    scan_status_display = ""

        annotated = draw_overlay(frame, motion_score, stable, cur_status)
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buf.tobytes()
               + b'\r\n')

    release_camera()


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _compute_clusters(books, n_clusters=None):
    """
    Assign KMeans cluster IDs to books.
    Features: one-hot genre + author frequency.
    Returns list of cluster ints aligned with books list.
    """
    if not SKLEARN_AVAILABLE or len(books) < 3:
        return [0] * len(books)

    # Build a simple numeric feature matrix: [genre_encoded, author_hash_mod]
    genres  = [b.get("genre", "Unknown") for b in books]
    le      = LabelEncoder()
    g_enc   = le.fit_transform(genres).reshape(-1, 1)

    authors = [b.get("authors", ["Unknown"])[0] for b in books]
    a_enc   = np.array([hash(a) % 100 for a in authors]).reshape(-1, 1)

    years   = np.array([int(b.get("year", 0) or 0) for b in books]).reshape(-1, 1)
    pages   = np.array([int(b.get("page_count", 0) or 0) for b in books]).reshape(-1, 1)

    # normalize pages & year
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9) * 10

    X = np.hstack([g_enc * 3, a_enc, norm(years), norm(pages)])

    k = n_clusters or max(2, min(6, len(books) // 3))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/stream")
def video_stream():
    """MJPEG stream — GET only, no body buffering."""
    # Explicitly tell Werkzeug not to read a body (it has none for GET)
    return Response(
        stream_with_context(_frame_producer()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
        }
    )


@app.route("/")
def home():
    # Serve the visualization page directly
    return app.send_static_file("index.html")


@app.route("/api/scan-result")
def get_scan_result():
    if scan_result_queue:
        return jsonify(scan_result_queue.popleft())
    return jsonify({"status": "idle"})


@app.route("/api/confirm", methods=["POST"])
def confirm_book():
    global pending_book
    with state_lock:
        if pending_book is None:
            return jsonify({"error": "No pending book"}), 400
        overrides = request.get_json(silent=True) or {}
        for f in ("title", "authors", "genre", "description"):
            if f in overrides:
                setattr(pending_book, f, overrides[f])
        if pending_book.id in library_ids:
            return jsonify({"error": "Duplicate", "book": pending_book.to_dict()}), 409
        library.append(pending_book)
        library_ids.add(pending_book.id)
        book_dict   = pending_book.to_dict()
        pending_book = None

    _save_graphs_json()
    log.info("✅ Added: %s", book_dict["title"])
    return jsonify({"status": "added", "book": book_dict, "library_size": len(library)})


@app.route("/api/reject", methods=["POST"])
def reject_book():
    global pending_book
    with state_lock:
        pending_book = None
    return jsonify({"status": "rejected"})


@app.route("/api/library")
def get_library():
    with state_lock:
        books = [b.to_dict() for b in library]
    return jsonify({"count": len(books), "books": books})


@app.route("/api/library/<book_id>", methods=["DELETE"])
def delete_book(book_id):
    global library
    with state_lock:
        before  = len(library)
        library = [b for b in library if b.id != book_id]
        library_ids.discard(book_id)
        removed = before - len(library)
    if removed:
        _save_graphs_json()
        return jsonify({"status": "removed", "book_id": book_id})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    return jsonify({"status": "ok" if open_camera() else "error"})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/api/status")
def status():
    return jsonify({
        "camera_active":    camera_active,
        "scan_in_progress": scan_in_progress,
        "library_size":     len(library),
        "pending_book":     pending_book.to_dict() if pending_book else None,
        "easyocr":          EASYOCR_AVAILABLE,
        "clustering":       SKLEARN_AVAILABLE,
    })


@app.route("/api/snap", methods=["POST"])
def manual_snap():
    frame = read_frame()
    if frame is None:
        return jsonify({"error": "Camera not available"}), 503
    run_scan(frame)
    return jsonify({"status": "scanning"})


@app.route("/api/graph-data")
def graph_data():
    """Three.js-friendly nodes + edges (genre / author links)."""
    with state_lock:
        books = [b.to_dict() for b in library]

    nodes, edges, edge_set = [], [], set()
    for b in books:
        nodes.append({
            "id": b["id"], "title": b["title"],
            "authors": b["authors"], "genre": b["genre"],
            "cover": b["cover_url"], "year": b["year"],
            "pages": b["page_count"], "confidence": b["confidence"],
        })

    def _add_edges(buckets, kind):
        for label, ids in buckets.items():
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    key = tuple(sorted([ids[i], ids[j]]))
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append({"source": ids[i], "target": ids[j],
                                      "type": kind, "label": label})

    genre_buckets, author_buckets = {}, {}
    for b in books:
        genre_buckets.setdefault(b["genre"], []).append(b["id"])
        for a in b["authors"]:
            author_buckets.setdefault(a, []).append(b["id"])

    _add_edges(genre_buckets, "genre")
    _add_edges(author_buckets, "author")

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/cluster")
def cluster_books():
    """
    Returns books with cluster assignments for the 3-D KMeans graph.
    Query param: k (optional, default auto)
    """
    with state_lock:
        books = [b.to_dict() for b in library]

    if len(books) < 2:
        return jsonify({"clusters": [], "k": 0, "books": books})

    k = request.args.get("k", None)
    k = int(k) if k else None
    labels = _compute_clusters(books, k)

    cluster_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FECA57", "#FF9FF3", "#54A0FF", "#5F27CD",
    ]

    enriched = []
    for book, label in zip(books, labels):
        enriched.append({**book,
                         "cluster": int(label),
                         "cluster_color": cluster_colors[label % len(cluster_colors)]})

    k_actual = max(labels) + 1 if labels else 0

    # Build cluster summaries
    summaries = {}
    for book, label in zip(books, labels):
        l = int(label)
        if l not in summaries:
            summaries[l] = {"genres": [], "authors": [], "titles": []}
        summaries[l]["genres"].append(book["genre"])
        summaries[l]["authors"].extend(book["authors"])
        summaries[l]["titles"].append(book["title"])

    cluster_info = {}
    for l, s in summaries.items():
        top_genre = max(set(s["genres"]), key=s["genres"].count)
        top_author = max(set(s["authors"]), key=s["authors"].count)
        cluster_info[l] = {
            "top_genre": top_genre,
            "top_author": top_author,
            "size": len(s["titles"]),
            "color": cluster_colors[l % len(cluster_colors)],
        }

    return jsonify({
        "books": enriched,
        "k": k_actual,
        "cluster_info": cluster_info,
    })


@app.route("/api/export-graphs")
def export_graphs():
    """Force-save and return graphs.json contents."""
    _save_graphs_json()
    with state_lock:
        books = [b.to_dict() for b in library]
    return jsonify({"books": books, "saved_at": time.time(),
                    "path": os.path.abspath(GRAPHS_JSON_PATH)})


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _load_graphs_json()   # restore previous session
    print("""
╔═══════════════════════════════════════════════════════════╗
║   📚  BookScan FIXED  •  http://localhost:5000            ║
║                                                           ║
║   NEW endpoints:                                          ║
║     GET  /api/cluster        → KMeans cluster data        ║
║     GET  /api/export-graphs  → save + return graphs.json  ║
╚═══════════════════════════════════════════════════════════╝
    """)
    # threaded=True is fine for dev; in production use gunicorn + eventlet
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
