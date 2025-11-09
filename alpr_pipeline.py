"""
This file contains the core computer vision and machine learning logic
for our ALPR (Automated License Plate Recognition) pipeline.

This is the NEW (v2) pipeline that uses a trained YOLOv8 model
and Tesseract-OCR.
"""

import cv2
import numpy as np
import base64
import pytesseract
from ultralytics import YOLO

# --- CONFIGURATION -----------------------------------------------------------

# Path to the 'best.pt' model file you trained.
# This assumes it's in the default 'runs/detect/train/weights/' directory.
# Update this path if your model is saved elsewhere.
YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'

# Tesseract OCR configuration.
# This config helps Tesseract recognize the format of a license plate.
# -l eng: Use the English language.
# --psm 8: Treat the image as a single word.
# --oem 3: Use the default OCR Engine Mode.
# -c tessedit_char_whitelist: Only allow these characters (reduces errors).
TESSERACT_CONFIG = r'-l eng --oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# --- HELPER FUNCTIONS --------------------------------------------------------

def encode_image(image):
    """Encodes a cv2 image (numpy array) into a base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def order_points(pts):
    """
    Sorts 4 corner points (x, y) into a consistent order:
    top-left, top-right, bottom-right, bottom-left.
    """
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum (x + y),
    # and the bottom-right will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference (y - x),
    # and the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective warp to an image to get a "bird's-eye view".
    'pts' must be the 4 ordered corner points from order_points().
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # or top-right and top-left x-coordinates.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between top-right and bottom-right
    # or top-left and bottom-left y-coordinates.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the warped image
    # (a simple rectangle). We use a 3:1 aspect ratio.
    # Standard license plates are ~2:1, but this gives padding.
    if maxWidth > maxHeight * 2:
        maxHeight = int(maxWidth / 2.5) # Adjust aspect ratio
    else:
        maxWidth = int(maxHeight * 2.5)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# --- YOLO MODEL LOADING ------------------------------------------------------

# We load the model once when the server starts to save time.
# This is much more efficient than loading it for every request.
try:
    print(f"Loading YOLOv8 model from {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Failed to load YOLO model. !!!")
    print(f"Error: {e}")
    print("Please ensure 'train.py' has run successfully and 'best.pt' exists.")
    model = None

# --- MAIN PIPELINE FUNCTION --------------------------------------------------

def run_ml_pipeline(image_data):
    """
    Runs the full ALPR pipeline on a single image.
    image_data: A numpy array (cv2 image).
    """
    
    # Create a dictionary to store the results
    results = {
        "step_1_gray": None,        # Grayscale image
        "step_2_ocr_prep": None,    # B&W warped plate
        "step_3_plate": None,       # Original image with bounding box
        "step_4_ocr": "[Error: OCR failed]" # Final text
    }

    # Make a copy of the original for drawing on later
    original_image = image_data.copy()

    # --- Step 1: Grayscale (Simple) ---
    # This is just for display on the webpage.
    # The ML model doesn't need it.
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    results["step_1_gray"] = encode_image(gray_image)

    # --- Step 2: YOLOv8 Plate Localization ---
    if model is None:
        print("Model is not loaded, skipping detection.")
        return results

    print("Running YOLOv8 detection...")
    # Run detection
    # We set conf=0.4 to be 40% confident.
    detections = model(image_data, conf=0.4, verbose=False)
    
    # Check if we found anything
    if not detections or len(detections[0].boxes) == 0:
        print("No license plate detected.")
        results["step_3_plate"] = encode_image(original_image)
        results["step_4_ocr"] = "[Error: No plate detected]"
        return results

    # Get the *first* detection (highest confidence)
    # This assumes one plate per image for this project.
    box = detections[0].boxes[0]
    
    # Get the bounding box coordinates (xyxy format)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    
    # Get the 4 corner points (if the model provides segmentation/keypoints)
    # Our model was trained for detection (boxes), not segmentation (corners).
    # So, we must *approximate* the 4 corners from the bounding box.
    # This is less accurate than a true segmentation model,
    # but works for this project.
    #
    # If your model was trained with segmentation, you'd use:
    # corners = box.xy[0].cpu().numpy().astype(int)
    #
    # But since it's a box, we define the corners from the box:
    corners = np.array([
        [x1, y1], # top-left
        [x2, y1], # top-right
        [x2, y2], # bottom-right
        [x1, y2]  # bottom-left
    ], dtype="float32")

    print(f"Plate detected at: [{x1}, {y1}, {x2}, {y2}]")

    # Draw the bounding box on the original image for the UI
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    results["step_3_plate"] = encode_image(original_image)

    # --- Step 3: Perspective Transform (Isolate Plate) ---
    # Use the 4 corners of the box to warp the image.
    try:
        warped_plate = four_point_transform(image_data, corners)
    except Exception as e:
        print(f"Error during perspective warp: {e}")
        # Fallback: simple crop (less accurate but won't crash)
        plate_crop = image_data[y1:y2, x1:x2]
        warped_plate = cv2.resize(plate_crop, (200, 70), interpolation=cv2.INTER_AREA)

    # --- Step 4: OCR Pre-processing & Recognition ---
    try:
        # Convert warped plate to grayscale
        ocr_gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold (Otsu's method) to get a clean B&W image
        # This is the single most important step for good OCR
        _, ocr_binary = cv2.threshold(ocr_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Invert the image if text is white (Tesseract prefers black text)
        if cv2.mean(ocr_binary)[0] > 127:
             ocr_binary = cv2.bitwise_not(ocr_binary)

        # Store the B&W image for the UI
        results["step_2_ocr_prep"] = encode_image(ocr_binary)

        # Run Tesseract OCR
        print("Running Tesseract OCR...")
        text = pytesseract.image_to_string(ocr_binary, config=TESSERACT_CONFIG)
        
        # Clean up the text (remove newlines, spaces, etc.)
        final_text = "".join(filter(str.isalnum, text)).upper()
        
        if not final_text:
             print("OCR returned empty string.")
             results["step_4_ocr"] = "[Error: OCR read no text]"
        else:
             print(f"OCR Result: '{final_text}'")
             results["step_4_ocr"] = final_text

    except Exception as e:
        print(f"Error during Tesseract OCR: {e}")
        # Send a blank B&W image if it fails
        results["step_2_ocr_prep"] = encode_image(np.zeros((70, 200), dtype="uint8"))
        results["step_4_ocr"] = "[Error: Tesseract failed]"

    return results