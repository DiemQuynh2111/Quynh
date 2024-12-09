import cv2
import numpy as np
import streamlit as st
import os
import gdown
import yt_dlp
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

# Thanh nhập số lượng vật thể để giám sát
monitor_counts = {}
for obj in object_names:
    monitor_counts[obj] = st.sidebar.number_input(f"Enter number of {obj} to monitor", min_value=0, value=0, step=1)

frame_limit = st.sidebar.slider("Set Frame Limit for Alarm", 1, 10, 3)

# Chọn nguồn video
video_source = st.radio("Choose Video Source", ["Upload File", "YouTube URL", "Custom Video URL (RTSP/Webcam)"])
temp_video_path = "temp_video.mp4"

# Nút điều khiển
start_button = st.button("Start Detection")
stop_button = st.button("Stop and Delete Video")

# Xử lý video nguồn
cap = None

# Nếu video đã được tải lên
if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        cap = cv2.VideoCapture(temp_video_path)

elif video_source == "YouTube URL":
    youtube_url = st.text_input("Paste YouTube URL here")
    if youtube_url:
        st.video(youtube_url)
        # Lấy video từ URL YouTube bằng yt_dlp hoặc thư viện khác
        # Tuy nhiên, với OpenCV, việc xử lý video trực tiếp từ YouTube yêu cầu chuyển đổi hoặc tải xuống trước
        # Thêm đoạn mã tải video từ YouTube nếu cần

elif video_source == "Custom Video URL (RTSP/Webcam)":
    video_url = st.text_input("Enter RTSP/Webcam URL")
    if video_url:
        cap = cv2.VideoCapture(video_url)

# Kiểm tra nếu video đã được tải lên hoặc từ cổng RTSP/Webcam
if cap is not None:
    ret, frame = cap.read()

    if start_button:
        st.session_state.running = True
        st.session_state.cap = cap
        st.session_state.start_time = time()

    if stop_button:
        st.session_state.running = False
        st.session_state.cap.release()
        st.session_state.cap = None
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.success("Video has been stopped and deleted.")

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
        detected_objects = {}

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

                    if label not in detected_objects:
                        detected_objects[label] = {'count': 1, 'time': time() - st.session_state.start_time}
                    else:
                        detected_objects[label]['count'] += 1
                        detected_objects[label]['time'] = time() - st.session_state.start_time

        # Hiển thị video đã xử lý
        st.image(frame, channels="BGR", use_container_width=True)

        # Cảnh báo khi vượt qua giới hạn
        for obj, info in detected_objects.items():
            if monitor_counts.get(obj, 0) > 0 and info['count'] >= monitor_counts[obj]:
                st.warning(f"ALERT: {obj} detected for {info['time']} seconds!")

else:
    st.info("Please upload a video or provide a YouTube URL or Custom Video URL (RTSP/Webcam).")
