# Background Removal API

This is a Flask-based web application that allows users to upload an image, remove its background using the **DeepLabV3 model**, and download the processed image with a white background. The application leverages deep learning models to perform semantic segmentation, specifically the **DeepLabV3 model** from **PyTorch**.

## Features
- **Image Upload**: Upload any image to the server.
- **Background Removal**: The DeepLabV3 model removes the background and keeps the foreground.
- **Processed Image**: View the processed image with a white background.
- **Download**: Option to download the processed image in **PNG** format.

## Technologies Used
- **Flask**: A lightweight Python web framework for creating the web application.
- **PyTorch**: A deep learning library for model inference (DeepLabV3 used for semantic segmentation).
- **OpenCV**: Used for image processing.
- **NumPy**: For array manipulations.
- **PIL (Pillow)**: For image manipulation and saving the final output.
- **Werkzeug**: Utility for secure file handling.

## Installation

### Prerequisites:
- **Python 3.6+**
- **pip** (Python package installer)

### Steps to set up the project:
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/background-removal-api.git
   cd background-removal-api

### Directory Structure:

```
background-removal-api/
│
├── app.py                # Main Flask application
├── static/
│   └── uploads/          # Directory to store uploaded and processed images
├── templates/
│   ├── index.html        # HTML template for the homepage
│
└── requirements.txt      # Python dependencies
```
