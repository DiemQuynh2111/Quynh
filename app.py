import cv2
import numpy as np
import streamlit as st
import os
import gdown
from time import time
import io

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

# Thêm Non-Maximum Suppression (NMS) để loại bỏ các bounding boxes chồng lấn
def apply_nms(boxes, confidences, threshold=0.4):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices

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
    lost_objects = set()  # Set để theo dõi các vật thể mất

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
        confidences = []
        class_ids = []

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
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Áp dụng NMS
        indices = apply_nms(boxes, confidences)

        for i in indices:
            i = i[0]
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]].lower()
            color = COLORS[class_ids[i]]

            # Vẽ bounding box và nhãn
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Đếm và theo dõi
            if label not in detected_objects:
                detected_objects[label] = 1
                lost_objects_time[label] = time()  # Lưu thời gian khi vật thể xuất hiện
                lost_objects.discard(label)  # Xóa vật thể từ danh sách mất
            else:
                detected_objects[label] += 1

        # Kiểm tra vật thể không khớp (không có đủ số lượng đã nhập)
        for obj in object_names:
            expected_count = monitor_counts[obj]
            detected_count = detected_objects.get(obj, 0)
            if detected_count != expected_count:
                if obj not in lost_objects_time or time() - lost_objects_time[obj] > 5:  # Cập nhật lại mỗi 5 giây
                    if obj not in lost_objects:
                        current_time = time()
                        elapsed_time = current_time - lost_objects_time[obj]
                        elapsed_time_str = f"{int(elapsed_time // 3600)}:{int((elapsed_time % 3600) // 60)}:{int(elapsed_time % 60)}"
                        st.warning(f"ALERT: {obj} detected {detected_count} times instead of {expected_count}.")
                        st.write(f"Time of mismatch: {elapsed_time_str}")
                        lost_objects_time[obj] = current_time  # Cập nhật thời gian khi không khớp
                        lost_objects.add(obj)  # Thêm vật thể vào danh sách đã mất

        # Hiển thị video
        stframe.image(frame, channels="BGR", use_container_width=True)

if stop_button:
    if cap:
        cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    st.success("Video stopped and temporary file deleted.")
