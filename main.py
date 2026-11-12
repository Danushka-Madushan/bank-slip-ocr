import io
import cv2
import numpy as np
import imagehash
from PIL import Image, ImageChops, ImageEnhance
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR

app = FastAPI(title="Bank Receipt Verification Engine")

# Initialize PaddleOCR (CPU mode, English)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def check_ela(image_bytes: bytes, quality: int = 90) -> float:
    """Computes Error Level Analysis and returns the mean difference score."""
    original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Save temporary image at specified compression quality
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    # Calculate difference
    diff = ImageChops.difference(original, compressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    # Scale difference
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    
    # Return average pixel difference across the image
    diff_np = np.array(diff)
    return float(np.mean(diff_np))

@app.post("/verify-receipt")
async def verify_receipt(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 1. Perceptual Hash
    img_pil = Image.open(io.BytesIO(contents))
    img_phash = str(imagehash.phash(img_pil))
    
    # 2. Forensics: Check ELA
    ela_score = check_ela(contents)
    is_suspicious_compression = ela_score > 35.0  # Calibrate threshold against your dataset

    # 3. Text Extraction with PaddleOCR
    img_cv = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result = ocr.ocr(img_cv, cls=True)
    
    extracted_lines = []
    low_confidence_elements = []
    
    if result and result[0]:
        for line in result[0]:
            text, conf = line[1][0], line[1][1]
            extracted_lines.append(text)
            if conf < 0.70:
                low_confidence_elements.append({"text": text, "confidence": conf})

    raw_text = " ".join(extracted_lines)

    # 4. Basic Classification Example (Sri Lankan Banks)
    detected_bank = "Unknown"
    if any(k in raw_text.upper() for k in ["FRIMI", "NATIONS TRUST"]):
        detected_bank = "FriMi"
    elif any(k in raw_text.upper() for k in ["FLASH", "COMMERCIAL BANK"]):
        detected_bank = "Commercial Bank Flash"
    elif any(k in raw_text.upper() for k in ["SAMPATH", "VISHWA"]):
        detected_bank = "Sampath Bank"
    elif any(k in raw_text.upper() for k in ["BOC", "BANK OF CEYLON"]):
        detected_bank = "Bank of Ceylon"

    return {
        "detected_bank": detected_bank,
        "phash": img_phash,
        "authenticity": {
            "ela_score": round(ela_score, 2),
            "flagged_tampering": is_suspicious_compression,
            "low_confidence_tokens": low_confidence_elements
        },
        "raw_extracted_text": extracted_lines
    }
