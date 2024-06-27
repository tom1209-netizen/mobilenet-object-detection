# Object Detection with MobileNetSSD

This Streamlit application performs object detection on images and videos using a pre-trained MobileNetSSD model. 
The app allows users to upload images or videos and displays the processed outputs with detected objects highlighted.

## Features

- **Image Upload**: Upload an image and detect objects within the image.
- **Video Upload**: Upload a video and detect objects frame by frame.
- **Confidence Threshold Adjustment**: Adjust the confidence threshold for object detection.
- **Real-time Processing**: Displays processed frames in real-time for uploaded videos.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/tom1209-netizen/mobilenet-object-detection.git
cd object-detection-app
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Install the required libraries
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
