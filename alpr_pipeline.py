import cv2
import numpy as np

# --- Step 1: Grayscale ---
def step_1_grayscale(image):
    """
    Converts an image to grayscale.
    OpenCV uses BGR format, so gray = 0.299*R + 0.587*G + 0.114*B
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# --- Step 2: Edge Detection ---
def step_2_edge_detection(gray_image):
    """
    Applies Gaussian Blur to reduce noise and then Canny for edge detection.
    """
    # Apply a blur to reduce noise
    # (5, 5) is the kernel size, 0 is sigmaX
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny edge detection
    # 50 and 150 are the min and max thresholds
    edges_image = cv2.Canny(blurred, 50, 150)
    
    # We return a BGR image so it can be encoded as a .jpg
    edges_image_bgr = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)
    return edges_image_bgr

# --- Step 3: Plate Localization ---
def step_3_localize_plate(original_image, edges_image):
    """
    Finds contours in the edge-detected image and filters them
    to find the one that most resembles a license plate.
    """
    # The Canny function in step 2 already returned a BGR image.
    # We need to convert it back to single-channel gray for findContours.
    edges_gray = cv2.cvtColor(edges_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the edge-detected image
    # cv2.RETR_TREE finds all contours
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal/vertical/diagonal segments
    contours, _ = cv2.findContours(edges_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and keep the top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    
    # Loop over the top 10 contours
    for c in contours:
        # Approximate the contour shape
        peri = cv2.arcLength(c, True)
        # 0.018 * peri is the 'epsilon' - max distance from contour to approximated contour
        # True means the shape is closed
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
        # If the approximated contour has 4 corners, it's a quadrilateral
        if len(approx) == 4:
            # Get the bounding box (x, y, width, height)
            (x, y, w, h) = cv2.boundingRect(approx)
            
            # Calculate the aspect ratio
            aspect_ratio = float(w) / h
            
            # Check if aspect ratio is within a reasonable range for a license plate
            # (e.g., 1.5 to 2.7 are common ratios)
            if aspect_ratio > 1.5 and aspect_ratio < 2.7:
                plate_contour = approx
                break # We found our plate
                
    # Create a copy of the original image to draw on
    localized_image = original_image.copy()
    
    # If we found a plate contour, draw it on the image
    if plate_contour is not None:
        cv2.drawContours(localized_image, [plate_contour], -1, (0, 255, 0), 3) # Draw in green

    # This function will also be used in the next step
    return localized_image, plate_contour

