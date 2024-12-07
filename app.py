import os
import cv2
import numpy as np
import streamlit as st
import gdown
import urllib.request
import time
import streamlit.components.v1 as components
import threading
import tempfile

# Đường dẫn đến tệp weights, config, và classes
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "coco.names"

# URL cho tệp config và classes
config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Kiểm tra và tải tệp weights từ Google Drive nếu không tồn tại
if not os.path.exists(weights_file):
    gdown.download("https://drive.google.com/uc?id=1pT0G-Mk9QIbjbOT4WTKEAf4TsmfG7jRD", weights_file, quiet=False)
    st.write(f"Đã tải xuống {weights_file}")

# Kiểm tra và tải tệp config nếu không tồn tại
if not os.path.exists(config_file):
    urllib.request.urlretrieve(config_url, config_file)
    st.write(f"Đã tải xuống {config_file}")

# Kiểm tra và tải tệp classes nếu không tồn tại
if not os.path.exists(classes_file):
    urllib.request.urlretrieve(classes_url, classes_file)
    st.write(f"Đã tải xuống {classes_file}")

# Đọc các lớp từ tệp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Tạo màu sắc ngẫu nhiên cho các lớp đối tượng
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_file, config_file)

# Lấy các layer output
def get_output_layers(net):
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1:
        return [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        return [layer_names[i[0] - 1] for i in unconnected_out_layers]

# Streamlit UI
st.title("Phát Hiện Đối Tượng với YOLO")

# Sidebar để nhập tên đối tượng và số lượng cần giám sát
object_names_input = st.sidebar.text_input('Nhập Tên Đối Tượng (cách nhau bằng dấu phẩy)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
frame_limit = st.sidebar.slider('Đặt Giới Hạn Khung Hình cho Cảnh Báo', 1, 10, 3)

# Nhập số lượng vật thể cần giám sát
object_counts_input = {}
for obj in object_names:
    object_counts_input[obj] = st.sidebar.number_input(f'Nhập số lượng {obj} cần giám sát', min_value=0, value=0, step=1)

# Đường dẫn đến tệp âm thanh cảnh báo (nếu cần)
alarm_sound = "police.wav"  # Đảm bảo tệp âm thanh này nằm cùng thư mục với app.py

# Thêm âm thanh cảnh báo vào ứng dụng Streamlit
def play_alarm():
    components.html(
        f"""
        <audio id="alarm-sound" src="{alarm_sound}" preload="auto"></audio>
        <script>
            function playAlarm() {{
                var sound = document.getElementById("alarm-sound");
                sound.play();
            }}
            window.playAlarm = playAlarm;
        </script>
        """,
        height=0,
    )

play_alarm()

# Khởi tạo session_state để lưu trữ trạng thái cảnh báo và thread
if 'alert' not in st.session_state:
    st.session_state.alert = False
if 'thread_running' not in st.session_state:
    st.session_state.thread_running = False

# Placeholder để hiển thị video
frame_placeholder = st.empty()

# Button để bắt đầu và dừng phát hiện
start_button = st.button("Bắt Đầu Phát Hiện")
stop_button = st.button("Dừng Phát Hiện")

# Hàm phát hiện đối tượng
def detect_objects(video_source):
    # Mở nguồn video (0 để mở webcam hoặc đường dẫn tới video)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error("Không thể mở nguồn video.")
        return

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể nhận khung hình từ nguồn video.")
            break

        height, width, channels = frame.shape

        # Phát hiện đối tượng
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        detected_objects = {obj: 0 for obj in object_names}

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id].lower() in object_names:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Áp dụng NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        missing_alert_triggered = False  # Đánh dấu nếu cần phát cảnh báo

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                color = COLORS[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Cập nhật số lượng vật thể đã phát hiện
                detected_objects[classes[class_ids[i]].lower()] += 1

        # Kiểm tra số lượng vật thể theo yêu cầu
        for obj in object_names:
            required_count = object_counts_input[obj]
            current_count = detected_objects[obj]

            if required_count > 0 and current_count < required_count:
                missing_object = obj
                cv2.putText(frame, f"Warning: {missing_object} Missing!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                missing_alert_triggered = True  # Kích hoạt cảnh báo
            elif current_count >= required_count:
                cv2.putText(frame, f"{obj.capitalize()}: {current_count}/{required_count}",
                            (50, 50 + object_names.index(obj) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Cập nhật trạng thái cảnh báo trong session_state
        if missing_alert_triggered:
            st.session_state.alert = True
        else:
            st.session_state.alert = False

        # Nếu cần, gọi hàm JavaScript để phát âm thanh cảnh báo
        if st.session_state.alert:
            st.session_state.thread_running = True
            play_alarm()

        # Hiển thị khung hình trong Streamlit
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# Chạy phát hiện đối tượng khi nhấn nút "Bắt Đầu Phát Hiện"
if start_button:
    st.session_state.running = True
    if not st.session_state.thread_running:
        threading.Thread(target=detect_objects, args=(0,)).start()

# Dừng phát hiện khi nhấn nút "Dừng Phát Hiện"
if stop_button:
    st.session_state.running = False
    st.session_state.thread_running = False
    st.session_state.alert = False
    frame_placeholder.empty()
