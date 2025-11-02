from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

# Import our pipeline functions
from alpr_pipeline import (
    step_1_grayscale, 
    step_2_edge_detection,
    step_3_localize_plate # <-- Import the new step
)

# Initialize the Flask app
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

def encode_image_to_base64(image):
    """Encodes an OpenCV image (Numpy array) to a base64 string"""
    # We encode as JPEG. PNG is also an option.
    _, buffer = cv2.imencode('.jpg', image)
    # Convert buffer to a byte string
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

@app.route('/process', methods=['POST'])
def process_image():
    """
    This is our API endpoint. It receives an image,
    runs the CV pipeline, and returns the processed images.
    """
    
    # Get the image file from the request
    file = request.files['image']
    
    # Read the image file into a numpy array
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    original_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if original_image is None:
        return jsonify({"error": "Could not decode image"}), 400

    print("Image received, running pipeline...")

    # --- Run Pipeline Steps ---
    
    # Step 1: Grayscale
    gray_image = step_1_grayscale(original_image)
    
    # Step 2: Edge Detection
    edges_image_bgr = step_2_edge_detection(gray_image) # This returns a BGR image
    
    # Step 3: Plate Localization
    # This step needs the original image (to draw on) and the edges image (to find contours)
    # It returns the image with the box drawn, and the contour coordinates for the next step
    localized_image, plate_contour = step_3_localize_plate(original_image, edges_image_bgr)
    
    # --- Encode Results ---
    
    # Grayscale image needs to be converted back to 3 channels for JPEG encoding
    gray_for_encoding = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    results = {
        "step_1_gray": encode_image_to_base64(gray_for_encoding),
        "step_2_edges": encode_image_to_base64(edges_image_bgr),
        "step_3_plate": encode_image_to_base64(localized_image), # <-- Send the new image
        "step_4_ocr": "..." # Placeholder
    }
    
    print("Pipeline finished, sending results.")
    return jsonify(results)

if __name__ == '__main__':
    # Run the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)

