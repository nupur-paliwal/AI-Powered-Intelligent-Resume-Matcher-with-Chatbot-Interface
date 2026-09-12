"""Extract text from resume PDFs, falling back to OCR for scanned pages."""
import logging

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

import config

logger = logging.getLogger(__name__)

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

# A page with fewer than this many characters of native text is treated
# as a scanned image and re-processed with OCR.
MIN_NATIVE_CHARS_PER_PAGE = 40
OCR_RENDER_ZOOM = 2.0  # ~144 DPI, a good balance of OCR accuracy vs. speed


class ExtractionError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


def extract_text(file_path):
    """Extract text from a PDF file.

    Returns a dict with:
        text: the full extracted text (all pages joined by blank lines)
        method: "native", "ocr", or "mixed"
        pages: number of pages processed
        warnings: list of human-readable warnings (e.g. OCR errors per page)
    """
    warnings = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # noqa: BLE001 - surfacing any parse failure
        raise ExtractionError(f"Could not open PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ExtractionError("PDF has no pages")

    page_texts = []
    used_native = False
    used_ocr = False

    for page_index in range(doc.page_count):
        page = doc[page_index]
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_NATIVE_CHARS_PER_PAGE:
            page_texts.append(native_text)
            used_native = True
            continue

        ocr_text, ocr_warning = _ocr_page(page, page_index)
        if ocr_warning:
            warnings.append(ocr_warning)
        if ocr_text.strip():
            page_texts.append(ocr_text.strip())
            used_ocr = True
        elif native_text:
            # OCR found nothing useful; keep whatever native text existed.
            page_texts.append(native_text)
            used_native = True

    page_count = doc.page_count
    doc.close()

    if used_native and used_ocr:
        method = "mixed"
    elif used_ocr:
        method = "ocr"
    else:
        method = "native"

    full_text = "\n\n".join(t for t in page_texts if t)
    if not full_text.strip():
        warnings.append(
            "No extractable text found even after OCR; the file may be "
            "empty, corrupted, or an unsupported image format."
        )

    return {
        "text": full_text,
        "method": method,
        "pages": page_count,
        "warnings": warnings,
    }


def _ocr_page(page, page_index):
    """Render a page to an image and run Tesseract OCR on it."""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_ZOOM, OCR_RENDER_ZOOM))
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(image)
        return text, None
    except pytesseract.TesseractNotFoundError:
        return "", (
            "Tesseract OCR is not installed or not on PATH; page "
            f"{page_index + 1} could not be OCR'd. Install tesseract-ocr "
            "and/or set TESSERACT_CMD in .env."
        )
    except Exception as exc:  # noqa: BLE001 - keep processing other pages
        logger.warning("OCR failed on page %d: %s", page_index + 1, exc)
        return "", f"OCR failed on page {page_index + 1}: {exc}"
