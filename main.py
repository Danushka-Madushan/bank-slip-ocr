"""
Bank Receipt Verification Engine (Sri Lankan bank / wallet receipts)
===================================================================

Produces a *risk assessment* for an uploaded receipt image. It cannot prove a
payment happened -- the only real proof is the credit appearing in your own
account / bank API / CEFT statement. Treat the output as triage for manual
review, not as an automatic accept/reject decision.

Install (Python 3.9+):
    pip install fastapi "uvicorn[standard]" python-multipart pillow imagehash \
        opencv-python-headless numpy python-dateutil paddlepaddle paddleocr

Run (the OCR model is loaded once per worker process):
    uvicorn receipt_engine:app --workers 1

Environment variables (all optional):
    MAX_UPLOAD_BYTES, MAX_PIXELS, MAX_RECEIPT_AGE_DAYS, RECEIPT_DB_PATH
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import cv2
import imagehash
import numpy as np
from dateutil import parser as date_parser
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from paddleocr import PaddleOCR
from PIL import ExifTags, Image, ImageChops, ImageOps

log = logging.getLogger("receipt-engine")
LK_TZ = timezone(timedelta(hours=5, minutes=30))  # Sri Lanka has no DST
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    max_pixels: int = int(os.getenv("MAX_PIXELS", 25_000_000))
    ocr_max_side: int = 1800          # larger images are downscaled for OCR
    ocr_min_side: int = 900           # smaller images are upscaled 2x for OCR
    ocr_low_conf: float = 0.70
    ela_quality: int = 90
    phash_max_distance: int = 4       # Hamming distance (of 64 bits) for "near duplicate"
    min_bank_score: int = 3
    max_receipt_age_days: int = int(os.getenv("MAX_RECEIPT_AGE_DAYS", 30))
    db_path: str = os.getenv("RECEIPT_DB_PATH", "receipts.db")


CFG = Config()


# --------------------------------------------------------------------------- #
# Risk signals
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    code: str
    weight: float   # 0 = informational; otherwise a rough "chance this indicates a problem"
    detail: str


def combine_risk(signals: list[Signal]) -> float:
    """Noisy-OR: independent signals compound but the score stays within 0-100."""
    p_ok = 1.0
    for s in signals:
        p_ok *= 1.0 - min(max(s.weight, 0.0), 0.99)
    return round((1.0 - p_ok) * 100, 1)


def risk_level(score: float) -> tuple[str, str]:
    if score < 20:
        return "low", "accept"
    if score < 50:
        return "medium", "manual_review"
    return "high", "escalate"


# --------------------------------------------------------------------------- #
# OCR wrapper (PaddleOCR 2.x and 3.x compatible, thread-safe)
# --------------------------------------------------------------------------- #
@dataclass
class OcrItem:
    text: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) in native-image pixels

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


def _bbox(points) -> tuple:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def parse_ocr_result(raw) -> list[OcrItem]:
    items: list[OcrItem] = []
    if not raw or raw[0] is None:
        return items
    page = raw[0]
    if isinstance(page, Mapping):  # PaddleOCR 3.x
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", page.get("dt_polys", []))
        for text, score, poly in zip(texts, scores, polys):
            items.append(OcrItem(str(text).strip(), float(score), _bbox(poly)))
    else:  # PaddleOCR 2.x: [[box, (text, conf)], ...]
        for line in page:
            (text, conf) = line[1]
            items.append(OcrItem(str(text).strip(), float(conf), _bbox(line[0])))
    return [i for i in items if i.text]


class OcrEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # Paddle predictors are not safe to share across threads
        self._engine = self._build()

    @staticmethod
    def _build():
        # Constructor kwargs differ between PaddleOCR major versions; try each in turn.
        attempts = (
            dict(use_angle_cls=True, lang="en", show_log=False, det_limit_side_len=CFG.ocr_max_side),
            dict(lang="en", use_textline_orientation=True, text_det_limit_side_len=CFG.ocr_max_side),
            dict(lang="en"),
        )
        last: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise RuntimeError("Could not initialise PaddleOCR") from last

    def read(self, bgr: np.ndarray) -> list[OcrItem]:
        with self._lock:
            try:
                raw = self._engine.ocr(bgr, cls=True)  # 2.x
            except TypeError:
                raw = self._engine.ocr(bgr)            # 3.x dropped `cls`
        return parse_ocr_result(raw)


# --------------------------------------------------------------------------- #
# Reading-order text with char-offset -> OCR-box mapping
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    items: list          # OcrItems in reading order
    text: str            # rows joined by "\n", boxes within a row joined by " "
    spans: list          # (start, end, item_index) per item

    @classmethod
    def build(cls, items: list) -> "Document":
        if not items:
            return cls([], "", [])
        med_h = max(float(np.median([max(i.height, 1) for i in items])), 1.0)
        rows: list[list[OcrItem]] = []
        row_y = 0.0
        for it in sorted(items, key=lambda i: i.yc):
            if rows and abs(it.yc - row_y) <= 0.6 * med_h:
                rows[-1].append(it)
                row_y = float(np.mean([i.yc for i in rows[-1]]))
            else:
                rows.append([it])
                row_y = it.yc

        parts: list[str] = []
        spans: list[tuple] = []
        ordered: list[OcrItem] = []
        pos = 0
        for row in rows:
            for it in sorted(row, key=lambda i: i.bbox[0]):
                spans.append((pos, pos + len(it.text), len(ordered)))
                ordered.append(it)
                parts.append(it.text)
                pos += len(it.text)
                parts.append(" ")
                pos += 1
            parts[-1] = "\n"  # same length, so offsets stay valid
        return cls(ordered, "".join(parts), spans)

    def items_in(self, start: int, end: int) -> list:
        return [self.items[i] for s, e, i in self.spans if s < end and e > start]

    def line_context(self, start: int, end: int) -> str:
        """Label words on the same line as a value (or the previous line if the value is alone)."""
        ls = self.text.rfind("\n", 0, start) + 1
        le = self.text.find("\n", end)
        le = len(self.text) if le == -1 else le
        own = f"{self.text[ls:start]} {self.text[end:le]}"
        letters = re.sub(r"[^A-Za-z]", "", re.sub(r"\b(?:LKR|RS)\b\.?", "", own, flags=re.I))
        if len(letters) < 3 and ls > 0:
            ps = self.text.rfind("\n", 0, ls - 1) + 1
            own = f"{self.text[ps:ls - 1]} {own}"
        return own.upper()


@dataclass
class Field:
    value: Any
    raw: str
    items: list

    @property
    def confidence(self) -> float:
        return round(min((i.conf for i in self.items), default=0.0), 3)

    def public(self) -> dict:
        v = self.value
        if isinstance(v, Decimal):
            v = f"{v:.2f}"
        elif isinstance(v, datetime):
            v = v.isoformat(timespec="minutes")
        return {"value": v, "raw": self.raw, "confidence": self.confidence}


# --------------------------------------------------------------------------- #
# Field extraction
# --------------------------------------------------------------------------- #
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
AMOUNT_PATTERNS = (
    re.compile(r"(?:\bLKR|\bRS\b\.?|රු\.?)\s*" + _NUM, re.I),   # LKR 5,000.00 / Rs.1,250
    re.compile(_NUM + r"\s*(?:LKR|RS)\b", re.I),                 # 5,000.00 LKR
    re.compile(r"\bAMOUNT\b[^\d\n]{0,20}" + _NUM, re.I),         # Amount: 5,000.00
)
AMOUNT_POSITIVE = ("AMOUNT", "TRANSFER", "PAID", "PAYMENT", "TOTAL", "SENT", "DEBIT")
AMOUNT_NEGATIVE = ("FEE", "CHARGE", "BALANCE", "AVAILABLE", "COMMISSION", "TAX", "LIMIT")


def extract_amount(doc: Document) -> tuple[Optional[Field], bool]:
    """Returns (best amount, whether another equally-ranked, different amount exists)."""
    seen: dict[int, tuple] = {}
    for rx in AMOUNT_PATTERNS:
        for m in rx.finditer(doc.text):
            try:
                val = Decimal(m.group(1).replace(",", ""))
            except InvalidOperation:
                continue
            if not (0 < val < 10**9) or m.start(1) in seen:
                continue
            ctx = doc.line_context(m.start(1), m.end(1))
            score = sum(w in ctx for w in AMOUNT_POSITIVE) - 2 * sum(w in ctx for w in AMOUNT_NEGATIVE)
            seen[m.start(1)] = (score, m.start(1), val, m)
    if not seen:
        return None, False
    ranked = sorted(seen.values(), key=lambda c: (-c[0], c[1]))
    best = ranked[0]
    ambiguous = any(c[0] == best[0] and c[2] != best[2] for c in ranked[1:])
    m = best[3]
    return Field(best[2], m.group(0).strip(), doc.items_in(m.start(1), m.end(1))), ambiguous


REF_LABEL = re.compile(
    r"\b(?:REF(?:ERENCE)?|TRANS(?:ACTION)?|TXN|TRACE|RRN|RECEIPT|CONFIRMATION)\b\s*(?:NO\.?|NUMBER|ID|#|CODE)?",
    re.I,
)
REF_TOKEN = re.compile(r"\b[A-Z0-9][A-Z0-9\-]{5,31}\b", re.I)
_DATEISH = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")


def extract_reference(doc: Document) -> Optional[Field]:
    for label in REF_LABEL.finditer(doc.text):
        for tok in REF_TOKEN.finditer(doc.text, label.end(), min(label.end() + 60, len(doc.text))):
            value = tok.group(0)
            if sum(c.isdigit() for c in value) < 4 or _DATEISH.match(value):
                continue
            if re.match(r"\.\d{1,2}\b", doc.text[tok.end():tok.end() + 3]):  # 125000.00 is an amount
                continue
            return Field(value.upper(), value, doc.items_in(tok.start(), tok.end()))
    return None


_TIME = r"(?:[,\s]+(?:AT\s+)?\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)?"
_MON = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*"
DATE_PATTERNS = (
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}" + _TIME, re.I),
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}" + _TIME, re.I),
    re.compile(r"\b\d{1,2}(?:ST|ND|RD|TH)?[\s\-]*" + _MON + r"[\s,\-]*\d{2,4}" + _TIME, re.I),
    re.compile(r"\b" + _MON + r"\s+\d{1,2}(?:ST|ND|RD|TH)?,?\s+\d{4}" + _TIME, re.I),
)


def extract_datetime(doc: Document, now: datetime) -> Optional[Field]:
    candidates: list[tuple[bool, Field]] = []
    for rx in DATE_PATTERNS:
        for m in rx.finditer(doc.text):
            raw = m.group(0).strip()
            iso_style = bool(re.match(r"\d{4}[-/]", raw))
            try:
                # dateutil swaps month/day for YYYY-MM-DD when dayfirst=True, so only use it for D/M/Y styles
                dt = date_parser.parse(raw, dayfirst=not iso_style)
            except (ValueError, OverflowError):
                continue
            if 2000 <= dt.year <= now.year + 1:
                candidates.append((bool(re.search(r"\d:\d{2}", raw)), Field(dt, raw, doc.items_in(m.start(), m.end()))))
    if not candidates:
        return None
    with_time = [c for c in candidates if c[0]]
    return (with_time or candidates)[0][1]


SUCCESS_RX = re.compile(r"\b(?:SUCCESS(?:FUL(?:LY)?)?|COMPLETED|APPROVED|TRANSFERRED)\b", re.I)
FAILURE_RX = re.compile(
    r"\b(?:FAILED|FAILURE|UNSUCCESSFUL|DECLINED|REJECTED|CANCELL?ED|REVERSED|PENDING|IN\s+PROGRESS)\b", re.I
)


def detect_status(doc: Document) -> Optional[str]:
    if m := FAILURE_RX.search(doc.text):
        return "pending" if re.match(r"PENDING|IN\s+PROGRESS", m.group(0), re.I) else "failed"
    if SUCCESS_RX.search(doc.text):
        return "success"
    return None


# --------------------------------------------------------------------------- #
# Bank detection (weighted, header-aware)
# --------------------------------------------------------------------------- #
_BANK_SIGNATURES = {
    "FriMi (Nations Trust Bank)": ((r"\bFRI\s?MI\b", 3), (r"NATIONS\s+TRUST", 3), (r"\bNTB\b", 1)),
    "Commercial Bank": ((r"COMMERCIAL\s+BANK", 3), (r"\bCOM\s?BANK\b", 3), (r"\bFLASH\b", 1)),
    "Sampath Bank": ((r"\bSAMPATH\b", 3), (r"\bVISHWA\b", 2)),
    "Bank of Ceylon": ((r"BANK\s+OF\s+CEYLON", 3), (r"\bBOC\b", 2)),
    "Hatton National Bank": ((r"HATTON\s+NATIONAL", 3), (r"\bHNB\b", 3)),
    "People's Bank": ((r"PEOPLE'?S\s+BANK", 3), (r"PEOPLE'?S\s+WAVE", 2)),
    "Seylan Bank": ((r"\bSEYLAN\b", 3),),
    "NDB Bank": ((r"NATIONAL\s+DEVELOPMENT\s+BANK", 3), (r"\bNDB\b", 3)),
    "DFCC Bank": ((r"\bDFCC\b", 3),),
    "Pan Asia Bank": ((r"PAN\s+ASIA", 3),),
    "Union Bank": ((r"UNION\s+BANK", 3),),
    "Cargills Bank": ((r"CARGILLS\s+BANK", 3),),
    "Amana Bank": ((r"\bAMANA\b", 3),),
}
_BANK_RX = {b: [(re.compile(p, re.I), w) for p, w in rules] for b, rules in _BANK_SIGNATURES.items()}


def detect_bank(doc: Document, image_height: int) -> dict:
    """Matches in the top quarter of the receipt count double, so a beneficiary-bank
    line further down can't outvote the issuing bank's header/branding."""
    scores: dict[str, int] = {}
    for it in doc.items:
        zone = 2 if it.yc < 0.25 * image_height else 1
        for bank, rules in _BANK_RX.items():
            for rx, weight in rules:
                if rx.search(it.text):
                    scores[bank] = scores.get(bank, 0) + weight * zone
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < CFG.min_bank_score:
        return {"name": "Unknown", "confidence": 0.0, "candidates": dict(ranked[:3])}
    best = ranked[0][1]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    return {"name": ranked[0][0], "confidence": round(best / (best + runner_up), 2), "candidates": dict(ranked[:3])}


# --------------------------------------------------------------------------- #
# Image loading + forensics
# --------------------------------------------------------------------------- #
EDITOR_MARKERS = (
    "photoshop", "gimp", "canva", "picsart", "pixlr", "photopea", "paint.net", "affinity",
    "illustrator", "figma", "inkscape", "fotor", "picmonkey", "snapseed", "lightroom", "ibis paint",
)
_XMP_TOOL_RX = re.compile(r'(?:CreatorTool|softwareAgent)(?:="|>)([^"<]{1,200})', re.I)


def _xmp_packet(raw: bytes) -> str:
    start = raw.find(b"<x:xmpmeta")
    if start == -1:
        return ""
    end = raw.find(b"</x:xmpmeta>", start)
    return raw[start:(end + 12 if end != -1 else start + 65536)].decode("utf-8", "ignore")


def analyze_metadata(img: Image.Image, raw: bytes) -> dict:
    exif = img.getexif()
    tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    software = str(tags.get("Software", "")).strip()
    raw_png_software = img.info.get("Software")
    png_software = raw_png_software.strip() if isinstance(raw_png_software, str) else ""
    tools: list[str] = [str(t) for t in _XMP_TOOL_RX.findall(_xmp_packet(raw))]
    text_elements: list[str] = [software, png_software, *tools]
    haystack = " ".join(text_elements).lower()
    return {
        "format": img.format,
        "width": img.width,
        "height": img.height,
        "orientation": exif.get(0x0112),
        "software": software or png_software or None,
        "xmp_tools": tools[:5],
        "editor_markers": sorted(m for m in EDITOR_MARKERS if m in haystack),
        "has_exif": bool(exif),
        "has_camera_exif": bool(tags.get("Make") or tags.get("Model")),
    }


def load_image(contents: bytes) -> tuple[Image.Image, dict]:
    try:
        img = Image.open(io.BytesIO(contents))
        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(415, f"Unsupported image format {img.format!r}; use JPEG, PNG or WEBP.")
        if img.width * img.height > CFG.max_pixels:
            raise HTTPException(413, "Image resolution too large.")
        meta = analyze_metadata(img, contents)
        img.load()
    except HTTPException:
        raise
    except Exception as exc:  # UnidentifiedImageError, truncated/corrupt files, decompression bombs
        raise HTTPException(422, "Could not decode the uploaded image.") from exc
    img = ImageOps.exif_transpose(img)
    rgba = img.convert("RGBA")  # flatten transparency onto white so OCR/ELA see what a human sees
    flat = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    return flat, meta


def prepare_for_ocr(img: Image.Image) -> tuple[Image.Image, float]:
    longest = max(img.size)
    if longest > CFG.ocr_max_side:
        scale = CFG.ocr_max_side / longest
    elif longest < CFG.ocr_min_side:
        scale = 2.0
    else:
        return img, 1.0
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS), scale


def compute_ela(img: Image.Image, quality: int) -> np.ndarray:
    """Per-pixel Error Level Analysis map (un-normalised, so values are comparable across regions).
    Must run at native resolution: resizing destroys the 8x8 JPEG block grid ELA depends on."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as resaved:
        diff = ImageChops.difference(img, resaved.convert("RGB"))
    return np.asarray(diff).max(axis=2).astype(np.float32)


def ela_numeric_outliers(ela: np.ndarray, items: list) -> list:
    """Flags numeric text boxes whose ELA is far above that of similar-sized boxes on the
    same receipt (pasted/retyped digits often carry a different compression history).
    Comparing against peers -- not the whole image -- avoids the white-background problem
    that makes a global ELA threshold meaningless on receipts."""
    h, w = ela.shape
    stats = []
    for it in items:
        x1, y1, x2, y2 = it.bbox
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            stats.append((it, y2 - y1, float(ela[y1:y2, x1:x2].mean()), (x1, y1, x2, y2)))

    flagged = []
    for it, height, val, box in stats:
        if not any(c.isdigit() for c in it.text):
            continue
        peers = np.array([v for o, hh, v, _ in stats if o is not it and 0.7 <= hh / height <= 1.3])
        if len(peers) < 5:
            continue  # nothing comparable -- can't judge (e.g. a unique oversized amount)
        med = float(np.median(peers))
        mad = float(np.median(np.abs(peers - med)))
        sigma = max(1.4826 * mad, 0.15 * med, 0.25)
        z = (val - med) / sigma
        if z >= 4.0 and val >= 1.8 * med:
            flagged.append((it, {"text": it.text, "bbox": box, "ela_mean": round(val, 2),
                                 "peer_median": round(med, 2), "z": round(z, 1)}))
    flagged.sort(key=lambda f: -f[1]["z"])
    return flagged[:8]


# --------------------------------------------------------------------------- #
# Duplicate store (SQLite; swap for Postgres + a BK-tree/pgvector at scale)
# --------------------------------------------------------------------------- #
def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class ReceiptStore:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.create_function("hamming_hex", 2, _hamming_hex, deterministic=True)
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS receipts (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, order_id TEXT,
                sha256 TEXT NOT NULL, phash TEXT NOT NULL,
                bank TEXT, reference TEXT, amount TEXT, receipt_dt TEXT, risk_score REAL
            );
            CREATE INDEX IF NOT EXISTS ix_receipts_sha ON receipts(sha256);
            CREATE INDEX IF NOT EXISTS ix_receipts_ref ON receipts(reference);
            """
        )

    def check_and_register(self, rec: dict, *, register: bool, max_distance: int) -> list[dict]:
        """Atomically look for prior matches, then (optionally) record this receipt."""
        with self._lock:
            rows = self._db.execute(
                """SELECT * FROM receipts
                   WHERE sha256 = :sha256
                      OR (reference IS NOT NULL AND reference = :reference
                          AND (bank IS NULL OR :bank IS NULL OR bank = :bank))
                      OR hamming_hex(phash, :phash) <= :dist
                   ORDER BY created_at DESC LIMIT 20""",
                {**rec, "dist": max_distance},
            ).fetchall()
            if register:
                self._db.execute(
                    """INSERT INTO receipts (id, created_at, order_id, sha256, phash, bank, reference, amount, receipt_dt)
                       VALUES (:id, :created_at, :order_id, :sha256, :phash, :bank, :reference, :amount, :receipt_dt)""",
                    rec,
                )
                self._db.commit()
        return [dict(r) for r in rows]

    def set_risk(self, receipt_id: str, score: float) -> None:
        with self._lock:
            self._db.execute("UPDATE receipts SET risk_score = ? WHERE id = ?", (score, receipt_id))
            self._db.commit()

    def close(self) -> None:
        self._db.close()


def duplicate_signals(matches: list[dict], rec: dict) -> list[Signal]:
    out: list[Signal] = []
    for m in matches[:5]:
        where = f"receipt {m['id'][:8]} ({m['created_at']})"
        if rec["order_id"] and m["order_id"] == rec["order_id"]:
            out.append(Signal("resubmission_same_order", 0.0, f"Already submitted for order {rec['order_id']}: {where}."))
            continue
        same_ref = bool(rec["reference"]) and m["reference"] == rec["reference"]
        if m["sha256"] == rec["sha256"]:
            out.append(Signal("exact_duplicate_file", 0.80, f"Byte-identical file already seen: {where}."))
        elif same_ref and m["amount"] and rec["amount"] and m["amount"] != rec["amount"]:
            out.append(Signal("reference_reused_different_amount", 0.90,
                              f"Same reference as {where} but amount differs ({m['amount']} vs {rec['amount']})."))
        elif same_ref:
            out.append(Signal("duplicate_reference", 0.75, f"Same bank reference already used: {where}."))
        elif not (rec["reference"] and m["reference"]):
            # Receipts from one app share a layout, so pHash alone is weak evidence. Only treat it
            # as a duplicate when a reference number can't tell the two apart.
            out.append(Signal("near_duplicate_image", 0.40, f"Visually near-identical to {where}."))
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def analyze_receipt(
    contents: bytes,
    *,
    ocr: OcrEngine,
    store: ReceiptStore,
    order_id: Optional[str],
    expected_amount: Optional[Decimal],
    register: bool,
    include_raw_text: bool,
) -> dict:
    t0 = time.perf_counter()
    img, meta = load_image(contents)
    sha256 = hashlib.sha256(contents).hexdigest()
    phash = str(imagehash.phash(img))
    now = datetime.now(LK_TZ).replace(tzinfo=None)
    signals: list[Signal] = []

    # ---- OCR (on a resized copy; boxes are mapped back to native pixels) ----
    ocr_img, scale = prepare_for_ocr(img)
    items = ocr.read(cv2.cvtColor(np.asarray(ocr_img), cv2.COLOR_RGB2BGR))
    for it in items:
        it.bbox = tuple(int(round(v / scale)) for v in it.bbox)
    doc = Document.build(items)

    # ---- Field extraction ----
    bank = detect_bank(doc, img.height)
    amount, amount_ambiguous = extract_amount(doc)
    reference = extract_reference(doc)
    when = extract_datetime(doc, now)
    status = detect_status(doc)
    fields = {"amount": amount, "reference": reference, "datetime": when}

    # ---- Content / consistency signals ----
    if not items:
        signals.append(Signal("no_text_detected", 0.30, "OCR found no text; image may not be a receipt or is unreadable."))
    else:
        missing = [n for n, f in fields.items() if f is None]
        if missing:
            signals.append(Signal("missing_expected_fields", min(0.10 * len(missing), 0.30),
                                  f"Could not extract: {', '.join(missing)}."))
        weak = [n for n, f in fields.items() if f and f.confidence < CFG.ocr_low_conf]
        if weak:
            signals.append(Signal("low_confidence_field", 0.15, f"Low OCR confidence on: {', '.join(weak)}."))
    if bank["name"] == "Unknown" and items:
        signals.append(Signal("unrecognized_bank_template", 0.10, "No known bank/wallet identified."))
    if amount_ambiguous:
        signals.append(Signal("ambiguous_amount", 0.15, "Several different amounts scored equally; amount may be mis-picked."))

    if status in ("failed", "pending"):
        signals.append(Signal(f"status_{status}", 0.70 if status == "failed" else 0.50,
                              f"Receipt shows a {status} transaction."))
    elif items and status is None:
        signals.append(Signal("no_success_status", 0.10, "No explicit success/completed status found."))

    if when:
        dt = when.value
        if dt - now > timedelta(hours=1):
            signals.append(Signal("future_dated", 0.80, f"Receipt date {dt:%Y-%m-%d %H:%M} is in the future."))
        elif now - dt > timedelta(days=CFG.max_receipt_age_days):
            signals.append(Signal("stale_receipt", 0.30, f"Receipt is older than {CFG.max_receipt_age_days} days."))

    if expected_amount is not None:
        if amount is None:
            signals.append(Signal("amount_not_extracted", 0.30, "Expected amount supplied but none could be read."))
        elif amount.value != expected_amount:
            signals.append(Signal("amount_mismatch", 0.85,
                                  f"Receipt shows {amount.value:.2f}, expected {expected_amount:.2f}."))
        else:
            signals.append(Signal("amount_matches_expected", 0.0, "Receipt amount equals the expected amount."))

    # ---- Metadata ----
    if meta["editor_markers"]:
        signals.append(Signal("editing_software_metadata", 0.40,
                              f"Metadata names image-editing software: {', '.join(meta['editor_markers'])}."))

    # ---- ELA (weak on screenshots/PNG, so down-weighted unless the source is a JPEG) ----
    ela_info: dict = {"computed": False}
    if meta["orientation"] in (None, 1):  # rotation breaks the JPEG block grid ELA relies on
        ela = compute_ela(img, CFG.ela_quality)
        factor = 1.0 if meta["format"] == "JPEG" else 0.5
        outliers = ela_numeric_outliers(ela, items)
        key_ids = {id(i) for f in fields.values() if f for i in f.items}
        on_key = [o for o in outliers if id(o[0]) in key_ids]
        if on_key:
            signals.append(Signal("ela_outlier_on_key_field", 0.40 * factor,
                                  "Compression artefacts on amount/reference/date differ from comparable text."))
        elif outliers:
            signals.append(Signal("ela_outlier_numeric_text", 0.20 * factor,
                                  "Some numeric text has unusual compression artefacts vs. comparable text."))
        ela_info = {
            "computed": True,
            "source_format": meta["format"],
            "reliable_source": meta["format"] == "JPEG",
            "mean": round(float(ela.mean()), 3),
            "p99": round(float(np.percentile(ela, 99)), 3),
            "numeric_text_outliers": [o[1] for o in outliers],
        }
    else:
        ela_info["skipped_reason"] = f"EXIF orientation {meta['orientation']} (image was rotated)"

    # ---- Duplicate detection ----
    ref_key = re.sub(r"[^A-Z0-9]", "", reference.value) if reference else None
    rec = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "order_id": order_id,
        "sha256": sha256,
        "phash": phash,
        "bank": bank["name"] if bank["name"] != "Unknown" else None,
        "reference": ref_key,
        "amount": f"{amount.value:.2f}" if amount else None,
        "receipt_dt": when.value.isoformat() if when else None,
    }
    matches = store.check_and_register(rec, register=register, max_distance=CFG.phash_max_distance)
    signals.extend(duplicate_signals(matches, rec))

    # ---- Verdict ----
    score = combine_risk(signals)
    level, recommendation = risk_level(score)
    if register:
        store.set_risk(rec["id"], score)

    low_conf = [{"text": i.text, "confidence": round(i.conf, 3)} for i in items if i.conf < CFG.ocr_low_conf]
    ocr_out: dict = {
        "line_count": len(items),
        "mean_confidence": round(float(np.mean([i.conf for i in items])), 3) if items else None,
        "low_confidence_count": len(low_conf),
    }
    if include_raw_text:  # receipts contain names/account numbers, so this is opt-in
        ocr_out["text_lines"] = doc.text.split("\n") if doc.text else []
        ocr_out["low_confidence_tokens"] = low_conf

    return {
        "receipt_id": rec["id"],
        "registered": register,
        "risk": {"score": score, "level": level, "recommendation": recommendation},
        "signals": [{"code": s.code, "weight": s.weight, "detail": s.detail}
                    for s in sorted(signals, key=lambda s: -s.weight)],
        "bank": bank,
        "fields": {**{k: (f.public() if f else None) for k, f in fields.items()}, "status": status},
        "hashes": {"sha256": sha256, "phash": phash},
        "forensics": {"metadata": meta, "ela": ela_info},
        "ocr": ocr_out,
        "processing_ms": round((time.perf_counter() - t0) * 1000),
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    app.state.ocr = await run_in_threadpool(OcrEngine)
    app.state.store = ReceiptStore(CFG.db_path)
    yield
    app.state.store.close()


app = FastAPI(title="Bank Receipt Verification Engine", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/verify-receipt")
async def verify_receipt(
    request: Request,
    file: UploadFile = File(...),
    order_id: Optional[str] = Form(None, max_length=64),
    expected_amount: Optional[Decimal] = Form(None, gt=0),
    register: bool = Form(True),
    include_raw_text: bool = Form(False),
):
    """
    order_id         Your order/invoice id. Resubmitting the same receipt for the same order is
                     not treated as fraud; the same receipt on a *different* order is.
    expected_amount  If given, the extracted amount must match exactly.
    register         False = dry run (nothing is stored, so it can't trip duplicate checks later).
    include_raw_text Return OCR text (may contain PII). Off by default.
    """
    contents = await file.read(CFG.max_upload_bytes + 1)
    if not contents:
        raise HTTPException(400, "Empty file.")
    if len(contents) > CFG.max_upload_bytes:
        raise HTTPException(413, f"File exceeds {CFG.max_upload_bytes // (1024 * 1024)} MB limit.")

    result = await run_in_threadpool(
        analyze_receipt,
        contents,
        ocr=request.app.state.ocr,
        store=request.app.state.store,
        order_id=order_id,
        expected_amount=expected_amount,
        register=register,
        include_raw_text=include_raw_text,
    )
    log.info("receipt=%s risk=%s bank=%s ms=%s", result["receipt_id"][:8], result["risk"]["level"],
             result["bank"]["name"], result["processing_ms"])
    return result
