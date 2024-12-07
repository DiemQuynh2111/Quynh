import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

# Đường dẫn đến tệp weights, config và class names
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "coco.names"

# Kiểm tra và tải các tệp weights, config và class nếu chưa có
# Bạn cần đảm bảo các tệp này đã được tải về trước khi chạy ứng dụng
if not os.path.exists(weights_file):  
    st.error(f"File {weights_file} không tồn tại.")
if not os.path.exists(config_file):
    st.error(f"File {config_file} không tồn tại.")
if not os.path.exists(classes_file):
    st.error(f"File {classes_file} không tồn tại.")

# Đọc các lớp từ tệp coco.names
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
    return [layer_names[i - 1] for i in unconnected_out_layers]

# Hàm phát hiện đối tượng
def detect_objects_from_frame(frame):
    height, width, channels = frame.shape

    # Phát hiện đối tượng
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

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
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Áp dụng NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return indices, boxes, class_ids

# WebRTC callback function to handle webcam stream
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Chuyển đổi frame từ WebRTC (hình ảnh dạng YUV) sang BGR (OpenCV yêu cầu)
    img = frame.to_ndarray(format="bgr24")

    # Nhận diện đối tượng trên frame
    indices, boxes, class_ids = detect_objects_from_frame(img)
    
    # Kiểm tra đối tượng và hiển thị cảnh báo nếu cần thiết
    object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
    object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
    object_counts_input = {obj: st.sidebar.number_input(f'Enter number of {obj} to monitor', min_value=0, value=0, step=1) for obj in object_names}

    detected_objects = {obj: 0 for obj in object_names}

    for i in indices.flatten():
        box = boxes[i]
        class_id = class_ids[i]
        label = str(classes[class_id]).lower()
        
        if label in object_names:
            detected_objects[label] += 1

    # Hiển thị cảnh báo nếu số lượng vật thể chưa đủ
    for obj in object_names:
        required_count = object_counts_input[obj]
        current_count = detected_objects[obj]
        
        if required_count > 0 and current_count < required_count:
            cv2.putText(img, f"Warning: {obj} Missing!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_count >= required_count:
            cv2.putText(img, f"{obj.capitalize()}: {current_count}/{required_count}", (50, 50 + object_names.index(obj) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Trả về frame đã xử lý
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Phát Hiện Đối Tượng với YOLO và Webcam")

# Bắt đầu Streamlit WebRTC
webrtc_streamer(key="object-detection", video_frame_callback=video_frame_callback)
