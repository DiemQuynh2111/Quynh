import cv2
import numpy as np
import streamlit as st
import os
import gdown
from time import time
import datetime

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
lost_objects_time = {}
alert_flags = {}  # Cờ theo dõi trạng thái cảnh báo cho mỗi vật thể
for obj in object_names:
    monitor_counts[obj] = st.sidebar.number_input(f"Enter number of {obj} to monitor", min_value=0, value=0, step=1)

frame_limit = st.sidebar.slider("Set Frame Limit for Alarm", 1, 10, 3)

# Chọn nguồn video
video_source = st.radio("Choose Video Source", ["Upload File"])
temp_video_path = "temp_video.mp4"

# Nút điều khiển
start_button = st.button("Start Detection")
stop_button = st.button("Stop and Delete Video")

cap = None  # Biến để lưu nguồn video

# Âm thanh cảnh báo trực tiếp (sử dụng Streamlit)
alarm_audio = """
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
"""

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
                            alert_flags[label] = False  # Chưa cảnh báo
                        else:
                            detected_objects[label] += 1

                        # Cảnh báo
                        if detected_objects[label] > monitor_counts[label] and not alert_flags[label]:
                            st.warning(f"ALERT: {label} detected more than {monitor_counts[label]} times!")
                            alert_flags[label] = True  # Đánh dấu đã cảnh báo

        # Kiểm tra vật thể mất
        for obj in object_names:
            if obj not in detected_objects or detected_objects[obj] == 0:
                # Nếu vật thể không xuất hiện, kiểm tra thời gian đã trôi qua
                if obj not in lost_objects_time:  # Kiểm tra xem vật thể có trong lost_objects_time chưa
                    lost_objects_time[obj] = 0  # Nếu chưa có, khởi tạo thời gian mất là 0
                if time() - lost_objects_time[obj] > 5:  # 5 giây không phát hiện lại
                    # Cảnh báo chỉ một lần
                    if not alert_flags.get(obj, False):
                        st.warning(f"ALERT: {obj} not detected!")
                        st.markdown(alarm_audio, unsafe_allow_html=True)  # Phát âm thanh khi vật thể mất
                        alert_flags[obj] = True  # Đánh dấu đã cảnh báo

                    # Tính thời gian mất và hiển thị giờ, phút, giây
                    time_lost = time() - lost_objects_time[obj]
                    formatted_time = str(datetime.timedelta(seconds=int(time_lost)))
                    st.write(f"Time lost: {formatted_time}")  # Hiển thị thời gian mất

        # Hiển thị video
        stframe.image(frame, channels="BGR", use_container_width=True)

if stop_button:
    if cap:
        cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    st.success("Video stopped and temporary file deleted.")
