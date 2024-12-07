import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import gdown

# Check and download YOLO weights if not present
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    st.write("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Check for configuration files
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Configuration or classes file not found. Please check!")
    st.stop()

# Load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# Function to get output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Object detection function
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        color = COLORS[class_ids[i]]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        if frame is None:
            return None
        try:
            frame = cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_BGR2RGB)
            processed_frame = detect_objects(frame)
            return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return frame.to_ndarray()

# Streamlit UI
st.title("Object Detection with YOLO")

# WebRTC configuration
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

# Initialize video stream
try:
    webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False}
    )
except Exception as e:
    st.error(f"An error occurred while setting up the WebRTC stream: {e}")
