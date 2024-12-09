import cv2
import numpy as np
import streamlit as st
import os
import gdown
import yt_dlp
import pygame
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

# Hàm lấy layer đầu ra
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Khởi tạo pygame để chơi âm thanh
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("police.wav")  # Đảm bảo file police.wav tồn tại trong thư mục dự án

# Giao diện Streamlit
st.title("Object Detection with YOLO")

# Thanh bên trái để nhập thông tin
st.sidebar.header("Settings")
object_names_input = st.sidebar.text_input("Enter Object Names (comma separated)", "cell phone,laptop,umbrella")
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
monitor_counts = {}
lost_objects_time = {}
for obj in object_names:
    monitor_counts[obj] = st.sidebar.number_input(f"Enter number of {obj} to monitor", min_value=0, value=0, step=1)

frame_limit = st.sidebar.slider("Set Frame Limit for Alarm", 1, 10, 3)

# Chọn nguồn video
video_source = st.radio("Choose Video Source", ["Upload File", "YouTube URL"])
temp_video_path = "temp_video.mp4"

# Nút điều khiển
start_button = st.button("Start Detection")
stop_button = st.button("Stop and Delete Video")

cap = None  # Biến để lưu nguồn video

# Xử lý video từ nguồn
if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        cap = cv2.VideoCapture(temp_video_path)

elif video_source == "YouTube URL":
    youtube_url = st.text_input("Paste YouTube URL here")
    if youtube_url and start_button:
        ydl_opts = {'outtmpl': temp_video_path, 'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        st.success("YouTube video downloaded!")
        cap = cv2.VideoCapture(temp_video_path)

# Kiểm tra nếu có video để xử lý
if cap is not None and start_button:
    stframe = st.empty()
    detected_objects = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or no frames available.")
            break

        # Phát hiện vật thể
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        height, width, _ = frame.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = classes[class_id].lower()
                    if label in object_names:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2
                        color = COLORS[class_id]

                        # Vẽ khung và nhãn
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Đếm và theo dõi
                        if label not in detected_objects:
                            detected_objects[label] = 1
                            lost_objects_time[label] = time()  # Lưu thời gian khi vật thể xuất hiện
                        else:
                            detected_objects[label] += 1

                        # Cảnh báo
                        if detected_objects[label] > monitor_counts[label]:
                            st.warning(f"ALERT: {label} detected more than {monitor_counts[label]} times!")

        # Kiểm tra vật thể mất
        for obj in object_names:
            if obj not in detected_objects or detected_objects[obj] == 0:
                if obj not in lost_objects_time or time() - lost_objects_time[obj] > 5:  # 5 giây không phát hiện lại
                    st.warning(f"ALERT: {obj} not detected!")
                    alarm_sound.play()  # Phát âm thanh khi vật thể mất
                    lost_objects_time[obj] = time()  # Cập nhật lại thời gian mất vật thể

        # Hiển thị video
        stframe.image(frame, channels="BGR", use_container_width=True)

if stop_button:
    if cap:
        cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    st.success("Video stopped and temporary file deleted.")
