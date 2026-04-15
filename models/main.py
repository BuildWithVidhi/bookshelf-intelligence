"""
📚 BookScan Core — main.py
Flask backend with:
  - Live camera feed (MJPEG stream)
  - Stability detection (auto-snap)
  - OCR (Tesseract / EasyOCR fallback)
  - Book identification (Google Books API)
  - Duplicate prevention
  - REST API for frontend consumption

Install deps:
    pip install flask flask-cors opencv-python pytesseract easyocr requests numpy Pillow

Also install Tesseract binary:
    Ubuntu: sudo apt install tesseract-ocr
    Mac:    brew install tesseract
    Win:    https://github.com/UB-Mannheim/tesseract/wiki
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
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from PIL import Image
import io
import base64
from dataclasses import dataclass, asdict
from typing import Optional
from collections import deque

# ─── Optional: EasyOCR as fallback (GPU-accelerated if CUDA available) ───────
try:
    import easyocr
    EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR loaded (fallback OCR ready)")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("ℹ️  EasyOCR not installed — using Tesseract only")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bookscan")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/stream": {"origins": "*"}})

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0          # 0 = default webcam
STABILITY_FRAMES   = 12         # consecutive stable frames needed to auto-snap
STABILITY_THRESH   = 1.8        # mean pixel diff threshold (lower = stricter)
DETECT_BOX_PADDING = 0.12       # fraction of frame edge as padding
MIN_TEXT_LENGTH    = 4          # ignore OCR results shorter than this
GOOGLE_BOOKS_URL   = "https://www.googleapis.com/books/v1/volumes"
FRAME_WIDTH        = 1280
FRAME_HEIGHT       = 720
MAX_LIBRARY_SIZE   = 500


# ─── Data Model ───────────────────────────────────────────────────────────────
@dataclass
class Book:
    id: str
    title: str
    authors: list[str]
    genre: str
    description: str
    cover_url: str
    publisher: str
    year: str
    page_count: int
    language: str
    isbn: str
    raw_ocr: str
    confidence: float          # 0-1 match confidence
    added_at: float            # unix timestamp

    def to_dict(self):
        return asdict(self)


# ─── Global State ─────────────────────────────────────────────────────────────
camera_lock    = threading.Lock()
state_lock     = threading.Lock()
cap: Optional[cv2.VideoCapture] = None
camera_active  = False
library: list[Book] = []
library_ids: set[str] = set()          # for O(1) duplicate check
pending_book: Optional[Book] = None    # last detected, not yet confirmed
scan_result_queue = deque(maxlen=1)    # latest scan for SSE push

# stability ring buffer
frame_buffer: deque = deque(maxlen=STABILITY_FRAMES)
stable_count   = 0
last_snap_time = 0.0
SNAP_COOLDOWN  = 3.0   # seconds between auto-snaps


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def open_camera():
    global cap, camera_active
    with camera_lock:
        if cap is not None and cap.isOpened():
            return True
        log.info("📷 Opening camera...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize latency
        if not cap.isOpened():
            log.error("❌ Cannot open camera")
            camera_active = False
            return False
        camera_active = True
        log.info("✅ Camera opened")
        return True


def release_camera():
    global cap, camera_active
    with camera_lock:
        camera_active = False
        if cap is not None:
            cap.release()
            cap = None
            log.info("📷 Camera released")


def read_frame() -> Optional[np.ndarray]:
    """Thread-safe frame read."""
    with camera_lock:
        if cap is None or not cap.isOpened():
            return None
        ret, frame = cap.read()
        return frame if ret else None


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING & STABILITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_detect_box(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the center detection region."""
    h, w = frame.shape[:2]
    pad_x = int(w * DETECT_BOX_PADDING)
    pad_y = int(h * DETECT_BOX_PADDING)
    return pad_x, pad_y, w - pad_x, h - pad_y


def crop_to_detect_box(frame: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = get_detect_box(frame)
    return frame[y1:y2, x1:x2]


def is_stable(frame: np.ndarray) -> tuple[bool, float]:
    """
    Detect if the camera/book is steady by comparing consecutive frames.
    Returns (stable: bool, motion_score: float)
    """
    gray = cv2.cvtColor(crop_to_detect_box(frame), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    frame_buffer.append(gray)
    if len(frame_buffer) < 3:
        return False, 999.0

    diffs = []
    for i in range(1, len(frame_buffer)):
        diff = cv2.absdiff(frame_buffer[i], frame_buffer[i-1])
        diffs.append(np.mean(diff))

    motion_score = float(np.mean(diffs[-3:]))   # look at last 3 diffs
    return motion_score < STABILITY_THRESH, motion_score


def enhance_for_ocr(frame: np.ndarray) -> np.ndarray:
    """
    Pipeline: crop → denoise → sharpen → adaptive threshold → deskew
    Returns a high-contrast grayscale image optimized for Tesseract.
    """
    roi = crop_to_detect_box(frame)

    # upscale to improve OCR accuracy on small text
    scale = 2.0
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # fast non-local means denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # unsharp mask for sharpening
    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)

    # adaptive threshold — handles varying lighting on book covers
    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=12
    )

    # deskew
    thresh = deskew(thresh)
    return thresh


def deskew(img: np.ndarray) -> np.ndarray:
    """Straighten slightly tilted text using projection profile."""
    try:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 15:   # don't deskew if too extreme (probably wrong)
            return img
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


def draw_overlay(frame: np.ndarray, motion_score: float, stable: bool,
                 scan_status: str = "") -> np.ndarray:
    """Draw the detection box and status HUD on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_detect_box(frame)

    # semi-transparent dark mask outside the box
    mask = np.zeros_like(frame)
    mask[:y1, :] = 60
    mask[y2:, :] = 60
    mask[y1:y2, :x1] = 60
    mask[y1:y2, x2:] = 60
    overlay = cv2.addWeighted(overlay, 1.0, mask, 0.5, 0)

    # detection box color: green = stable, yellow = moving
    color = (0, 255, 100) if stable else (0, 200, 255)
    thickness = 3

    # draw rounded-corner box using arcs
    corner = 20
    cv2.line(overlay, (x1+corner, y1), (x2-corner, y1), color, thickness)
    cv2.line(overlay, (x1+corner, y2), (x2-corner, y2), color, thickness)
    cv2.line(overlay, (x1, y1+corner), (x1, y2-corner), color, thickness)
    cv2.line(overlay, (x2, y1+corner), (x2, y2-corner), color, thickness)
    cv2.ellipse(overlay, (x1+corner, y1+corner), (corner, corner), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-corner, y1+corner), (corner, corner), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1+corner, y2-corner), (corner, corner), 90,  0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-corner, y2-corner), (corner, corner), 0,   0, 90, color, thickness)

    # motion bar
    bar_max   = 200
    bar_h     = 12
    bar_y     = y2 + 18
    bar_fill  = int(min(motion_score / 5.0, 1.0) * bar_max)
    bar_color = (0, 255, 100) if stable else (0, 150, 255)
    cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_max, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(overlay, f"Motion: {motion_score:.2f}", (x1 + bar_max + 10, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # status text
    status_label = scan_status if scan_status else ("🔒 STEADY — scanning..." if stable else "🎯 Align book in box")
    cv2.putText(overlay, status_label, (x1, y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # library count
    cv2.putText(overlay, f"Library: {len(library)} books",
                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return overlay


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tesseract(img: np.ndarray) -> str:
    """Run Tesseract with book-cover optimized config."""
    config = (
        "--oem 3 "          # LSTM engine
        "--psm 3 "          # fully automatic page segmentation
        "-l eng "
        "--dpi 300 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :.,'-&!"
    )
    try:
        pil_img = Image.fromarray(img)
        text = pytesseract.image_to_string(pil_img, config=config)
        return text.strip()
    except Exception as e:
        log.warning(f"Tesseract error: {e}")
        return ""


def run_easyocr(img: np.ndarray) -> str:
    """EasyOCR on the original colour ROI — often better on stylized covers."""
    if not EASYOCR_AVAILABLE:
        return ""
    try:
        results = EASYOCR_READER.readtext(img, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as e:
        log.warning(f"EasyOCR error: {e}")
        return ""


def extract_text(frame: np.ndarray) -> str:
    """
    Run both OCR engines and merge results.
    Tesseract on enhanced grayscale + EasyOCR on original colour crop.
    """
    enhanced  = enhance_for_ocr(frame)
    colour_roi = crop_to_detect_box(frame)
    colour_roi = cv2.resize(colour_roi, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_LANCZOS4)

    tess_text = run_tesseract(enhanced)
    easy_text = run_easyocr(colour_roi)

    # merge: prefer longer result, deduplicate lines
    combined = "\n".join(filter(None, [tess_text, easy_text]))
    lines = []
    seen = set()
    for line in combined.splitlines():
        line = line.strip()
        key  = re.sub(r'\s+', '', line.lower())
        if len(line) >= MIN_TEXT_LENGTH and key not in seen:
            seen.add(key)
            lines.append(line)
    result = "\n".join(lines)
    log.info(f"OCR result ({len(result)} chars):\n{result[:300]}")
    return result


def clean_query(text: str) -> str:
    """
    Extract the most likely title + author tokens from raw OCR dump.
    Heuristic: take the first 2-3 non-trivial lines, strip noise.
    """
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    # Remove lines that look like prices, barcodes, purely numeric, etc.
    lines = [l for l in lines if not re.fullmatch(r'[\d\s\$\.\-]+', l)]
    # Take top lines — they usually contain title/author on a book cover
    query = " ".join(lines[:4])
    query = re.sub(r'[^\w\s\-\.\']', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query[:120]   # API limit


# ═══════════════════════════════════════════════════════════════════════════════
#  BOOK IDENTIFICATION — Google Books API
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "fiction": "Fiction", "novel": "Fiction", "thriller": "Thriller",
    "mystery": "Mystery", "science fiction": "Sci-Fi", "sci-fi": "Sci-Fi",
    "fantasy": "Fantasy", "biography": "Biography", "history": "History",
    "self-help": "Self-Help", "business": "Business", "economics": "Economics",
    "psychology": "Psychology", "philosophy": "Philosophy",
    "science": "Science", "technology": "Technology",
    "computers": "Technology", "programming": "Technology",
    "cooking": "Cooking", "art": "Art", "poetry": "Poetry",
    "children": "Children", "young adult": "Young Adult",
    "romance": "Romance", "horror": "Horror", "travel": "Travel",
    "health": "Health", "religion": "Religion", "sports": "Sports",
}


def map_genre(categories: list[str]) -> str:
    combined = " ".join(categories).lower()
    for key, label in CATEGORY_MAP.items():
        if key in combined:
            return label
    return categories[0].title() if categories else "Unknown"


def identify_book(ocr_text: str) -> Optional[Book]:
    """
    Query Google Books API with the OCR text and return the best match.
    Falls back to progressively shorter queries if no result.
    """
    query = clean_query(ocr_text)
    if not query or len(query) < 5:
        log.warning("OCR query too short — skipping API call")
        return None

    log.info(f"🔍 Google Books query: '{query}'")

    attempts = [
        query,
        " ".join(query.split()[:6]),   # shorter fallback
        " ".join(query.split()[:3]),   # last resort
    ]

    for attempt in attempts:
        if not attempt:
            continue
        try:
            params = {
                "q": attempt,
                "maxResults": 5,
                "printType": "books",
                "langRestrict": "en",
            }
            resp = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=6)
            resp.raise_for_status()
            data = resp.json()

            if data.get("totalItems", 0) == 0:
                log.info(f"No results for: '{attempt}'")
                continue

            items = data.get("items", [])
            # pick best item (first is usually best ranked by Google)
            for item in items:
                info = item.get("volumeInfo", {})
                title = info.get("title", "")
                if not title:
                    continue

                authors = info.get("authors", ["Unknown"])
                categories = info.get("categories", [])
                genre = map_genre(categories)

                # compute a simple confidence score
                ocr_lower = ocr_text.lower()
                confidence = 0.0
                for word in title.lower().split():
                    if len(word) > 3 and word in ocr_lower:
                        confidence += 0.2
                for author in authors:
                    for word in author.lower().split():
                        if len(word) > 3 and word in ocr_lower:
                            confidence += 0.15
                confidence = min(confidence, 1.0)

                # stable unique ID: hash of lowercase title + first author
                uid_source = f"{title.lower()}:{authors[0].lower()}"
                book_id = hashlib.md5(uid_source.encode()).hexdigest()[:12]

                image_links = info.get("imageLinks", {})
                cover = (
                    image_links.get("thumbnail") or
                    image_links.get("smallThumbnail") or ""
                ).replace("http://", "https://")   # force HTTPS

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
                    confidence  = round(confidence, 2),
                    added_at    = time.time(),
                )
                log.info(f"📚 Found: '{title}' by {authors} (confidence={confidence:.2f})")
                return book

        except requests.RequestException as e:
            log.warning(f"API error: {e}")
            break

    log.warning("❌ Could not identify book from OCR text")
    return None


def _extract_isbn(volume_info: dict) -> str:
    for id_entry in volume_info.get("industryIdentifiers", []):
        if id_entry.get("type") in ("ISBN_13", "ISBN_10"):
            return id_entry.get("identifier", "")
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAN PIPELINE (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════════

scan_in_progress = False


def run_scan(frame: np.ndarray):
    """OCR + identify in a background thread. Pushes result to queue."""
    global scan_in_progress, pending_book
    if scan_in_progress:
        return
    scan_in_progress = True

    def _worker():
        global scan_in_progress, pending_book
        try:
            log.info("⚡ Starting scan pipeline...")
            ocr_text = extract_text(frame)
            if len(ocr_text) < MIN_TEXT_LENGTH:
                log.warning("OCR returned too little text")
                return

            book = identify_book(ocr_text)
            if book:
                with state_lock:
                    # check duplicate
                    if book.id in library_ids:
                        log.info(f"⚠️  Duplicate skipped: {book.title}")
                        scan_result_queue.append({"status": "duplicate", "book": book.to_dict()})
                    else:
                        pending_book = book
                        scan_result_queue.append({"status": "found", "book": book.to_dict()})
            else:
                scan_result_queue.append({"status": "not_found", "ocr": ocr_text[:200]})
        except Exception as e:
            log.error(f"Scan pipeline error: {e}")
            scan_result_queue.append({"status": "error", "message": str(e)})
        finally:
            scan_in_progress = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
#  MJPEG STREAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

scan_status_display = ""
scan_status_lock    = threading.Lock()


def generate_frames():
    global stable_count, last_snap_time, scan_status_display

    if not open_camera():
        return

    while camera_active:
        frame = read_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # stability check
        stable, motion_score = is_stable(frame)

        # auto-snap logic
        now = time.time()
        should_snap = (
            stable and
            not scan_in_progress and
            (now - last_snap_time) > SNAP_COOLDOWN
        )

        with scan_status_lock:
            current_status = scan_status_display

        if should_snap:
            stable_count += 1
            if stable_count >= STABILITY_FRAMES:
                stable_count   = 0
                last_snap_time = now
                log.info("📸 Auto-snap triggered!")
                snap = frame.copy()
                run_scan(snap)
                with scan_status_lock:
                    scan_status_display = "⚡ Scanning..."
        else:
            if not stable:
                stable_count = 0
            with scan_status_lock:
                if scan_status_display == "⚡ Scanning..." and not scan_in_progress:
                    scan_status_display = ""

        # draw overlay
        annotated = draw_overlay(frame, motion_score, stable, current_status)

        # encode as JPEG
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

    release_camera()


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/stream")
def video_stream():
    """MJPEG camera stream."""
    return Response(
        stream_with_context(generate_frames()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
        }
    )


@app.route("/api/scan-result")
def get_scan_result():
    """
    Poll endpoint — returns latest scan result or null.
    Frontend polls this ~2x per second.
    """
    if scan_result_queue:
        result = scan_result_queue.popleft()
        return jsonify(result)
    return jsonify({"status": "idle"})


@app.route("/api/confirm", methods=["POST"])
def confirm_book():
    """
    User confirms the detected book → add to library.
    Optionally pass JSON with edited fields to override.
    """
    global pending_book
    with state_lock:
        if pending_book is None:
            return jsonify({"error": "No pending book"}), 400

        # allow frontend to patch fields
        overrides = request.get_json(silent=True) or {}
        for field in ("title", "authors", "genre", "description"):
            if field in overrides:
                setattr(pending_book, field, overrides[field])

        if pending_book.id in library_ids:
            return jsonify({"error": "Duplicate", "book": pending_book.to_dict()}), 409

        library.append(pending_book)
        library_ids.add(pending_book.id)
        book_dict = pending_book.to_dict()
        pending_book = None

    log.info(f"✅ Book added: {book_dict['title']}")
    return jsonify({"status": "added", "book": book_dict, "library_size": len(library)})


@app.route("/api/reject", methods=["POST"])
def reject_book():
    """User rejects the detected book (wrong scan)."""
    global pending_book
    with state_lock:
        pending_book = None
    return jsonify({"status": "rejected"})


@app.route("/api/library", methods=["GET"])
def get_library():
    """Return the full library as JSON."""
    with state_lock:
        books = [b.to_dict() for b in library]
    return jsonify({
        "count": len(books),
        "books": books,
    })


@app.route("/api/library/<book_id>", methods=["DELETE"])
def delete_book(book_id: str):
    """Remove a book from the library."""
    global library
    with state_lock:
        before = len(library)
        library = [b for b in library if b.id != book_id]
        library_ids.discard(book_id)
        removed = before - len(library)
    if removed:
        return jsonify({"status": "removed", "book_id": book_id})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    success = open_camera()
    return jsonify({"status": "ok" if success else "error"})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "camera_active":   camera_active,
        "scan_in_progress": scan_in_progress,
        "library_size":    len(library),
        "pending_book":    pending_book.to_dict() if pending_book else None,
        "easyocr":         EASYOCR_AVAILABLE,
    })


@app.route("/api/snap", methods=["POST"])
def manual_snap():
    """Trigger a manual scan (user pressed button)."""
    frame = read_frame()
    if frame is None:
        return jsonify({"error": "Camera not available"}), 503
    run_scan(frame)
    return jsonify({"status": "scanning"})


@app.route("/api/graph-data", methods=["GET"])
def graph_data():
    """
    Return library data in a Three.js-friendly graph format.
    Nodes = books, edges = shared genre/author.
    """
    with state_lock:
        books = [b.to_dict() for b in library]

    nodes = []
    edges = []
    edge_set = set()

    for b in books:
        nodes.append({
            "id":      b["id"],
            "title":   b["title"],
            "authors": b["authors"],
            "genre":   b["genre"],
            "cover":   b["cover_url"],
            "year":    b["year"],
            "pages":   b["page_count"],
        })

    # edges: same genre
    genre_buckets: dict[str, list[str]] = {}
    for b in books:
        genre_buckets.setdefault(b["genre"], []).append(b["id"])
    for genre, ids in genre_buckets.items():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                key = tuple(sorted([ids[i], ids[j]]))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({"source": ids[i], "target": ids[j], "type": "genre", "label": genre})

    # edges: shared author
    author_buckets: dict[str, list[str]] = {}
    for b in books:
        for author in b["authors"]:
            author_buckets.setdefault(author, []).append(b["id"])
    for author, ids in author_buckets.items():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                key = tuple(sorted([ids[i], ids[j]]))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({"source": ids[i], "target": ids[j], "type": "author", "label": author})

    return jsonify({"nodes": nodes, "edges": edges})


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   📚  BookScan Core Backend  •  http://localhost:5000 ║
║                                                      ║
║   Endpoints:                                         ║
║     GET  /stream              → MJPEG camera feed    ║
║     GET  /api/scan-result     → latest OCR result    ║
║     POST /api/snap            → manual scan trigger  ║
║     POST /api/confirm         → add book to library  ║
║     POST /api/reject          → discard detection    ║
║     GET  /api/library         → full library JSON    ║
║     GET  /api/graph-data      → Three.js graph data  ║
║     GET  /api/status          → system status        ║
╚══════════════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
