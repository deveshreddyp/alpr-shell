from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import traceback

# Import our pipeline functions
from alpr_pipeline import (
    step_3_localize_plate_yolo,
    step_4_ocr_and_prep
)

app = Flask(__name__)
CORS(app)  # This allows our webpage to talk to the server

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # --- Step 1: Read Image ---
        # Decode the base64 image
        img_data = base64.b64decode(data['image'])
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        
        # original_image is the main image we'll use for processing
        original_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # We also create a simple grayscale version for display
        gray_image_display = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # --- Step 3: Localize Plate (YOLO) ---
        # This function returns BOTH the localized image and the cropped plate
        # The 'cropped_plate' is just the small rectangle of the plate
        # The 'localized_image' is the full car image with the green box drawn on it
        localized_image, cropped_plate = step_3_localize_plate_yolo(original_image)
        
        # Prepare response object
        response_data = {
            'step_1_gray': encode_image(gray_image_display),
            'step_3_localized': None,     # Image with bounding box
            'step_2_ocr_prepped': None,   # B&W warped plate
            'step_4_ocr_text': None         # Final text
        }

        # --- Step 4 & 2: OCR and Pre-processing ---
        # We check if a plate was found in Step 3
        if cropped_plate is not None:
            # We now run OCR. This function returns:
            # 1. The final text
            # 2. The pre-processed image (thresholded) that was used for OCR
            ocr_text, ocr_prepped_image = step_4_ocr_and_prep(cropped_plate)
            
            # Update response with our new data
            response_data['step_3_localized'] = encode_image(localized_image)
            response_data['step_2_ocr_prepped'] = encode_image(ocr_prepped_image)
            response_data['step_4_ocr_text'] = ocr_text

        else:
            # Handle case where no plate was found
            # We still send back the grayscale image
            print("No plate detected by YOLO model.")
            response_data['step_3_localized'] = None # No localization image
            response_data['step_2_ocr_prepped'] = None # No pre-pped image
            response_data['step_4_ocr_text'] = "No Plate Found"

        return jsonify(response_data)

    except Exception as e:
        # Print the full error stack trace for debugging
        print(f"Error: {e}")
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500

def encode_image(image):
    """Encodes an OpenCV image (numpy array) to a base64 string for JSON."""
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    # debug=True will auto-reload the server when you save changes
    # Use host='0.0.0.0' to make it accessible on your network (optional)
    app.run(debug=True, port=5000)