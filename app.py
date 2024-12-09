import cv2
import numpy as np
import streamlit as st
import os
import gdown
from pytube import YouTube
import tempfile

# Kiểm tra và tải tệp yolov3.weights từ Google Drive nếu chưa tồn tại
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    st.write("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Kiểm tra và tải các tệp cấu hình
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Tệp cấu hình hoặc tệp classes không tồn tại. Vui lòng kiểm tra lại!")
    st.stop()

# Đọc các lớp từ tệp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except Exception as e:
    st.error(f"Lỗi khi tải mô hình YOLO: {e}")
    st.stop()

# Hàm lấy các lớp đầu ra từ YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Hàm phát hiện đối tượng
def detect_objects(frame, object_names, prev_objects):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    detected_objects = {obj: False for obj in object_names}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id].lower() in object_names:
                detected_objects[classes[class_id].lower()] = True

    return detected_objects

# Streamlit UI
st.title("Object Detection from Video (File Upload or YouTube)")

# Nhập đối tượng cần theo dõi
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]

# Chọn cách tải video: từ file hoặc YouTube URL
video_source = st.radio("Choose Video Source", ["Upload File", "YouTube URL"])

temp_video_path = "temp_video.mp4"

if video_source == "Upload File":
    # Tải video từ máy tính lên
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
elif video_source == "YouTube URL":
    # Tải video từ YouTube
    youtube_url = st.text_input("Paste YouTube URL here")
    if youtube_url:
        try:
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(file_extension="mp4", res="360p").first()
            if stream is not None:
                stream.download(filename=temp_video_path)
                st.success("YouTube video downloaded successfully!")
            else:
                st.error("Could not find a suitable stream for the video.")
        except Exception as e:
            st.error(f"Error downloading YouTube video: {e}")

# Xử lý video
if os.path.exists(temp_video_path):
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    # Theo dõi đối tượng mất
    prev_objects = {obj: True for obj in object_names}
    time_lost = {}

    st.video(temp_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_objects(frame, object_names, prev_objects)

        for obj, detected in detected_objects.items():
            if not detected and prev_objects[obj]:
                current_time = frame_count / fps
                time_lost[obj] = current_time
                st.warning(f"{obj.upper()} bị mất lúc: {int(current_time // 60)}:{int(current_time % 60)}")
                # Phát âm thanh cảnh báo
                st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg", format="audio/ogg")

            prev_objects[obj] = detected

        frame_count += 1

    cap.release()
    os.remove(temp_video_path)
else:
    st.info("Vui lòng tải video hoặc nhập URL để bắt đầu!")
