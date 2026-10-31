"""Turns a detected plate region into a validated Malaysian plate number."""
import re

import cv2

import config

# Malaysian standard plate: 1-3 letters, 1-4 digits, optional trailing letter (e.g. WA1234B, BJT8558)
PLATE_PATTERN = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$")
TWO_LINE_MAX_ASPECT = 1.7


def create_ocr():
    from paddleocr import PaddleOCR

    # The default medium models take ~3.4 s per plate on CPU; the mobile ones ~0.1 s,
    # fast enough to read several frames per car. Plate crops are small and upright,
    # so the document-level steps (page orientation, unwarping) are skipped too.
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        cpu_threads=config.OCR_THREADS,
    )


def crop_plate(frame, box, pad=0.1):
    """Cuts the plate out of the frame with a small margin, resized for OCR.

    YOLO boxes are often tight and clip the edge characters, hence the padding.
    The aspect ratio is kept so two-line plates (motorbikes, some cars) aren't squashed.
    Returns None for boxes too small to read.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 10 or y2 - y1 < 5:
        return None

    crop = frame[y1:y2, x1:x2]
    scale = config.OCR_WIDTH / crop.shape[1]
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_two_line(plate_crop):
    """Two-line plates are nearly square (width/height ~1.2); single-line ones are 2+."""
    if plate_crop is None:
        return False
    h, w = plate_crop.shape[:2]
    return w / h < TWO_LINE_MAX_ASPECT


def read_plate(ocr, plate_crop):
    """Runs OCR on a plate crop. Returns (plate_text, ocr_confidence) or None if no valid plate."""
    if is_two_line(plate_crop):
        return read_two_line(ocr, plate_crop) or read_lines(ocr, plate_crop)
    return read_lines(ocr, plate_crop)


def read_two_line(ocr, plate_crop):
    """PaddleOCR's text detector often finds only one line of a two-line plate,
    so read the top and bottom halves separately (with a small overlap) and join them."""
    h = plate_crop.shape[0]
    top = read_lines(ocr, plate_crop[: int(h * 0.55)], validate=False)
    bottom = read_lines(ocr, plate_crop[int(h * 0.45):], validate=False)
    if not top or not bottom:
        return None
    plate = top[0] + bottom[0]
    if not PLATE_PATTERN.match(plate):
        return None
    return plate, min(top[1], bottom[1])


def read_lines(ocr, plate_crop, validate=True):
    """OCRs every text line in the crop. With validate=False, returns the joined
    (normalized) text without checking it against the plate format."""
    results = list(ocr.predict(plate_crop))
    if not results:
        return None
    texts = results[0].get("rec_texts", [])
    scores = list(results[0].get("rec_scores", []))
    if not texts:
        return None
    if not validate:
        return normalize("".join(texts)), float(min(scores, default=0.0))

    # Two-line plates can come back as two text lines, so try them joined first,
    # then each line alone (in case one line is a sticker or dealer badge).
    candidates = [("".join(texts), min(scores, default=0.0))]
    candidates += [(t, s) for t, s in zip(texts, scores)]
    for text, score in candidates:
        plate = normalize(text)
        if PLATE_PATTERN.match(plate):
            return plate, float(score)
    return None
