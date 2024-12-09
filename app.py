import cv2
import numpy as np
import streamlit as st
import os
import gdown
from time import time

# Tải YOLO weights và config nếu chưa có
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(weights_file):
    gdown.download("https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm", weights_file, quiet=False)

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Missing YOLO config or classes file.")
    st.stop()

# Đọc danh sách các lớp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_file, config_file)

# Hàm xử lý layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Giao diện Streamlit
st.title("Object Detection with YOLO")

# Thanh bên trái để nhập thông tin
st.sidebar.header("Settings")
object_names_input = st.sidebar.text_input("Enter Object Names (comma separated)", "cell phone,laptop,umbrella")
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]

video_source = st.radio("Choose Video Source", ["Upload File", "YouTube URL", "Custom Video URL (RTSP/Webcam)"])
temp_video_path = "temp_video.mp4"

if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")

elif video_source == "Custom Video URL (RTSP/Webcam)":
    video_url = st.text_input("Enter RTSP/Webcam URL")
    if video_url:
        st.success("Custom video URL entered.")

# Kiểm tra nếu video đã được tải lên
if os.path.exists(temp_video_path) or video_source == "Custom Video URL":
    video_path = temp_video_path if video_source == "Upload File" else video_url
    cap = cv2.VideoCapture(video_path)

    start_button = st.button("Start Detection")
    stop_button = st.button("Stop and Show Report")

    if start_button:
        st.session_state.running = True
        st.session_state.cap = cap
        st.session_state.start_time = time()
        st.session_state.missing_objects = {}

    if stop_button:
        st.session_state.running = False
        st.session_state.cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.success("Video has been stopped.")
        # Hiển thị báo cáo vật thể bị mất
        st.subheader("Missing Objects Report")
        for obj, timestamp in st.session_state.missing_objects.items():
            st.write(f"Object '{obj}' was lost at {timestamp:.2f} seconds.")

    if 'running' in st.session_state and st.session_state.running:
        cap = st.session_state.cap
        ret, frame = cap.read()

        if not ret:
            st.warning("Video ended.")
            st.session_state.running = False
            cap.release()

        # Xử lý phát hiện vật thể
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        height, width, _ = frame.shape
        detected_objects = set()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in object_names:
                    label = str(classes[class_id])
                    color = COLORS[class_id]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detected_objects.add(label)

        # Cập nhật danh sách vật thể bị mất
        current_time = time() - st.session_state.start_time
        for obj in object_names:
            if obj in detected_objects:
                st.session_state.missing_objects.pop(obj, None)  # Xóa khỏi danh sách nếu đã phát hiện lại
            else:
                if obj not in st.session_state.missing_objects:
                    st.session_state.missing_objects[obj] = current_time

        # Hiển thị video đã xử lý
        st.image(frame, channels="BGR", use_container_width=True)

else:
    st.info("Please upload a video or provide a valid URL.")
