import cv2
import numpy as np
import streamlit as st
import os
import gdown
from time import time
import io
from datetime import timedelta

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

# Giao diện Streamlit
st.title("Object Detection with YOLO")

# Thanh bên trái để nhập thông tin
st.sidebar.header("Settings")
object_names_input = st.sidebar.text_input("Enter Object Names (comma separated)", "cell phone,laptop,umbrella")
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
monitor_counts = {}
missing_object_counter = {}  # Đếm số khung hình vật thể bị mất
for obj in object_names:
    monitor_counts[obj] = st.sidebar.number_input(f"Enter number of {obj} to monitor", min_value=0, value=0, step=1)

frame_limit = st.sidebar.slider("Set Frame Limit for Alarm (frames)", 1, 10, 3)

# Chọn nguồn video
video_source = st.radio("Choose Video Source", ["Upload File"])
temp_video_path = "temp_video.mp4"

# Nút điều khiển
start_button = st.button("Start Detection")
stop_button = st.button("Stop and Delete Video")

cap = None  # Biến để lưu nguồn video

# Đảm bảo thư mục lưu file âm thanh đã tồn tại
audio_file_path = "/mnt/data/police.wav"
audio_dir = os.path.dirname(audio_file_path)

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Tải file âm thanh cảnh báo từ Google Drive nếu chưa có
audio_file_url = "https://drive.google.com/uc?id=19tOeyYVLZKHD9ETU4HHecHNYwoIa4sWq"  # ID từ Google Drive

# Tải file âm thanh nếu chưa có
if not os.path.exists(audio_file_path):
    gdown.download(audio_file_url, audio_file_path, quiet=False)

# Đọc file âm thanh cảnh báo
def play_alert_sound():
    if os.path.exists(audio_file_path):
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/wav')

# Xử lý video từ nguồn
if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        cap = cv2.VideoCapture(temp_video_path)

# Kiểm tra nếu có video để xử lý
if cap is not None and start_button:
    stframe = st.empty()
    detected_objects = {}
    missing_object_counter = {obj: 0 for obj in object_names}  # Đặt lại bộ đếm khi bắt đầu
    alerted_objects = set()  # Để theo dõi các đối tượng đã cảnh báo
    start_time = time()

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
        boxes = []
        class_ids = []
        confidences = []
        detected_objects.clear()

        # Lấy thông tin từ các lớp đầu ra
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
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        # Áp dụng Non-Maximum Suppression để loại bỏ các bounding box chồng lấp
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]].lower()
                color = COLORS[class_ids[i]]

                # Vẽ bounding box và nhãn
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Đếm và theo dõi
                if label in detected_objects:
                    detected_objects[label] += 1
                else:
                    detected_objects[label] = 1

        # Kiểm tra vật thể thiếu và đã quay lại
        for obj in object_names:
            current_count = detected_objects.get(obj, 0)

            if current_count == 0:  # Đối tượng không xuất hiện trong khung hình
                missing_object_counter[obj] += 1
                if missing_object_counter[obj] >= frame_limit and obj not in alerted_objects:
                    alerted_objects.add(obj)
                    missing_duration = str(timedelta(seconds=int(time() - start_time)))  # Thời gian mất
                    st.warning(f"⚠️ ALERT: '{obj}' is missing for {missing_duration}!")
                    play_alert_sound()  # Phát âm thanh cảnh báo khi đối tượng bị mất
            else:  # Đối tượng xuất hiện trở lại
                if missing_object_counter[obj] > 0:
                    missing_object_counter[obj] = 0  # Đặt lại bộ đếm khi vật thể xuất hiện
                if obj in alerted_objects:
                    alerted_objects.remove(obj)  # Xóa cảnh báo khi đối tượng quay lại

        # Hiển thị video
        stframe.image(frame, channels="BGR", use_container_width=True)

if stop_button:
    if cap:
        cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    st.success("Video stopped and temporary file deleted.")
