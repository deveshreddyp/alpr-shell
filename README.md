Automated License Plate Recognition (ALPR) Pipeline

This is a simple web-based application that demonstrates the computer vision pipeline for Automated License Plate Recognition (ALPR).

It uses a Python backend (with Flask and OpenCV) to perform the image processing and a simple HTML/CSS/JavaScript frontend to provide a user interface for uploading an image and viewing the results of each step.

(Suggestion: Replace the placeholder above with a screenshot of your running application, like the one you sent me.)

Tech Stack

Backend:

Python 3

Flask: A micro web framework for running the web server (API).

OpenCV (cv2): The core computer vision library for all image processing.

Numpy: For numerical operations and handling image arrays.

Frontend:

HTML5: For the page structure.

Tailwind CSS: For all styling and layout.

JavaScript (ES6+): For handling file uploads, calling the API (fetch), and displaying results.

Features (Pipeline Steps)

Load Image: Upload any JPG or PNG image of a car.

Step 1: Grayscale Conversion: The image is converted to grayscale, a necessary prerequisite for many CV tasks.

Step 2: Edge Detection: A Gaussian blur is applied to reduce noise, and the Canny edge detection algorithm is used to find all edges in the image.

Step 3: Plate Localization: The app finds all contours (shapes) in the edge-detected image. It then filters these contours to find the one that most likely represents a license plate by checking its shape (must be a quadrilateral) and its aspect ratio (width/height). A green bounding box is drawn around the best candidate.

Step 4: Character Recognition (OCR): (Placeholder for future development).

Setup and Installation

Follow these steps to get the project running on your local machine.

Prerequisites

Python 3.7+

pip (Python package installer)

Installation

Download or Clone the Repository:
Get all the project files (app.py, alpr_pipeline.py, index.html, requirements.txt) and place them in a single folder (e.g., alpr-project).

Open Your Terminal:
Navigate into your project folder:

cd path/to/alpr-project


Create a Virtual Environment (Recommended):
This keeps your project's dependencies separate from your main Python installation.

python -m venv venv


Activate the Virtual Environment:

On Windows:

.\venv\Scripts\activate


On macOS / Linux:

source venv/bin/activate


(Your terminal prompt should now start with (venv).)

Install Dependencies:
Use pip to install all the required Python libraries.

pip install -r requirements.txt


How to Run the Application

This is a client-server application, so you need to run both parts.

1. Start the Backend Server

With your virtual environment still active, run the app.py script to start the Flask server:

python app.py


You should see output similar to:

 * Serving Flask app 'app'
 * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
(Press CTRL+C to quit)


CRITICAL: Leave this terminal window open! This is your server.

2. Open the Frontend Application

In your computer's file explorer (not the terminal), navigate to your alpr-project folder.

Double-click the index.html file.

This will open the application in your default web browser.

3. Use the App

Click "Choose File" and select an image of a car.

Click the "Run Python Pipeline" button.

The JavaScript on the page will send the image to your local Python server, which will process it and send back the results for each pipeline step.

File Structure

alpr-project/
│
├── venv/                 # The virtual environment folder
│
├── app.py                # The Python Flask server (API)
├── alpr_pipeline.py      # All OpenCV computer vision logic
├── index.html            # The frontend webpage (UI)
└── requirements.txt      # Python dependencies for pip
