import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Paths to the model files
MODEL = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


def annotate_image(image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def main():
    st.title('Object Detection for Images and Videos')

    # Sidebar for options
    st.sidebar.title("Options")
    option = st.sidebar.radio("Select Input Type", ("Image", "Video"))
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    if option == "Image":
        file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
        if file is not None:
            st.image(file, caption="Uploaded Image")
            image = Image.open(file)
            image = np.array(image)
            detections = process_image(image)
            processed_image = annotate_image(image, detections, confidence_threshold)
            st.image(processed_image, caption="Processed Image")

    elif option == "Video":
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                detections = process_image(frame)
                processed_frame = annotate_image(frame, detections, confidence_threshold)
                stframe.image(processed_frame, channels="BGR", use_column_width=True)
            video.release()


if __name__ == "__main__":
    main()