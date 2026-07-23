import cv2
from .inference import visual_inference_allowed

_warned = False


def read_text(image):
    global _warned
    if not visual_inference_allowed():
        return ""
    try:
        import pytesseract
    except Exception:
        return ""

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return " ".join(pytesseract.image_to_string(gray).split())
    except Exception as exc:
        if not _warned:
            print(f"OCR unavailable: {exc}. No clear text detected.")
            _warned = True
        return ""
