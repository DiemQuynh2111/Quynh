import cv2
import numpy as np
import streamlit as st
import os
import time

# Cài đặt YOLO weights và config nếu chưa có
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(weights_file):
    st.error("Weights file not found!")
    st.stop()

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Config or classes file not found!")
    st.stop()

# Đọc danh sách các lớp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_file, config_file)

# Hàm lấy tên các layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Giao diện Streamlit
st.title("Object Detection and Loss Tracking")

# Thanh bên
object_names_input = st.sidebar.text_input("Enter Object Names (comma separated)", "cell phone,laptop")
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
frame_limit = st.sidebar.slider("Set Frame Limit for Alarm", 1, 10, 3)

video_source = st.radio("Choose Video Source", ["Upload File", "Webcam"])
temp_video_path = "temp_video.mp4"

# Phát hiện vật thể bị mất
object_loss_times = {}

if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        video_path = temp_video_path
    else:
        video_path = None

elif video_source == "Webcam":
    video_path = 0  # Mở webcam

start_button = st.button("Start Detection")

if start_button and video_path is not None:
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    start_time = time.time()
    detected_objects = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or no frames available.")
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        current_objects = set()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = str(classes[class_id])
                    if label.lower() in object_names:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        color = COLORS[class_id]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        current_objects.add(label.lower())

        # Phát hiện vật thể bị mất
        for obj in detected_objects - current_objects:
            if obj not in object_loss_times:
                object_loss_times[obj] = time.time() - start_time

        detected_objects = current_objects

        # Hiển thị khung hình
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

    # Kết quả cuối cùng
    st.subheader("Object Loss Report")
    for obj, loss_time in object_loss_times.items():
        st.write(f"Object '{obj}' was lost at {loss_time:.2f} seconds.")

else:
    if video_path is None:
        st.info("Please upload a video file or select Webcam.")
