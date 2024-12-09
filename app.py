import cv2
import numpy as np
import streamlit as st
import os
import gdown
import yt_dlp  # Sử dụng yt-dlp thay vì pytube

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

# Chọn video source
video_source = st.radio("Choose Video Source", ["Upload File", "YouTube URL"])
temp_video_path = "temp_video.mp4"

# Tải video từ máy hoặc URL
if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")

elif video_source == "YouTube URL":
    youtube_url = st.text_input("Paste YouTube URL here")
    if youtube_url and st.button("Download Video"):
        try:
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',  # Chọn chất lượng video và âm thanh tốt nhất
                'outtmpl': temp_video_path,  # Đặt tên tệp video tải xuống
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            st.success("YouTube video downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading YouTube video: {e}")

# Kiểm tra sự tồn tại của video
if os.path.exists(temp_video_path):
    try:
        # Hiển thị video mà không sử dụng 'use_container_width'
        st.video(temp_video_path)

        # Thêm nút điều khiển Start và Stop
        col1, col2 = st.columns(2)
        start_button = col1.button("Start")
        stop_button = col2.button("Stop")

        if start_button:
            st.session_state.running = True
            st.session_state.cap = cv2.VideoCapture(temp_video_path)

            if not st.session_state.cap.isOpened():
                st.error("Unable to open video file.")
                st.session_state.running = False
                st.session_state.cap.release()

        if stop_button:
            st.session_state.running = False
            if 'cap' in st.session_state:
                st.session_state.cap.release()
                st.session_state.cap = None

        # Xử lý video khi nút Start được nhấn
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

            st.image(frame, channels="BGR", use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying video: {e}")
else:
    st.info("Please upload a video or provide a YouTube URL.")
