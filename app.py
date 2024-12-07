import os
import gdown
import numpy as np
import pygame
import streamlit as st
import cv2
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from PIL import Image

# Kiểm tra và tải tệp yolov3.weights từ Google Drive nếu chưa tồn tại
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    print("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Kiểm tra và tải các tệp cấu hình nếu chưa tồn tại
config_file = "yolov3.cfg"
if not os.path.exists(config_file):
    config_drive_url = "https://drive.google.com/uc?id=1TfzHjG43gxo3s9fXX3-B_iQntYdqOUoH"
    print("Downloading yolov3.cfg from Google Drive...")
    gdown.download(config_drive_url, config_file, quiet=False)

# Kiểm tra và tải tệp lớp (classes) nếu chưa tồn tại
classes_file = "yolov3.txt"
if not os.path.exists(classes_file):
    classes_drive_url = "https://drive.google.com/uc?id=1fvce49sV1zK6Pggg8t4vxM3mmBGd6dk2"
    print("Downloading yolov3.txt from Google Drive...")
    gdown.download(classes_drive_url, classes_file, quiet=False)

# Cài đặt âm thanh
pygame.mixer.init()
alarm_sound = r"D:\AI\Project\police.wav"  # Đảm bảo đường dẫn đến âm thanh là đúng

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
st.title("Object Detection with YOLO")
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
frame_limit = st.sidebar.slider('Set Frame Limit for Alarm', 1, 10, 3)

# Nhập số lượng vật thể cần giám sát
object_counts_input = {}
for obj in object_names:
    object_counts_input[obj] = st.sidebar.number_input(f'Enter number of {obj} to monitor', min_value=0, value=0, step=1)

# Định nghĩa class xử lý video
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.object_not_found_counter = {obj: 0 for obj in object_names}
        self.initial_objects_count = {obj: 0 for obj in object_names}

    def recv(self, frame):
        # Chuyển đổi frame từ video stream sang ảnh
        img = frame.to_ndarray(format="bgr24")
        height, width, channels = img.shape

        # Nhận diện đối tượng với YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        detected_objects = {obj: 0 for obj in object_names}  # Đếm các vật thể đã phát hiện

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

        # Áp dụng NMS (Non-Maximum Suppression) để loại bỏ các bounding boxes dư thừa
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        object_found = False
        if len(indices) > 0:
            object_found = True
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                color = COLORS[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Cập nhật số lượng vật thể đã phát hiện
                detected_objects[classes[class_ids[i]].lower()] += 1

        # Kiểm tra số lượng vật thể theo yêu cầu
        for obj in object_names:
            required_count = object_counts_input[obj]
            current_count = detected_objects[obj]
            
            # Nếu số lượng vật thể phát hiện ít hơn yêu cầu, hiển thị cảnh báo
            if required_count > 0 and current_count < required_count:
                missing_object = obj
                cv2.putText(img, f"Warning: {missing_object} Missing!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pygame.mixer.music.load(alarm_sound)
                pygame.mixer.music.play()
            elif current_count >= required_count:  # Nếu số lượng vật thể đủ
                cv2.putText(img, f"{obj.capitalize()}: {current_count}/{required_count}", (50, 50 + object_names.index(obj) * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Cập nhật số lượng vật thể đã phát hiện
            self.initial_objects_count[obj] = detected_objects[obj]

        # Chuyển đổi ảnh từ BGR sang RGB để Streamlit hiển thị
        return img

# Sử dụng streamlit_webrtc để hiển thị webcam
webrtc_streamer(key="object-detection", video_processor_factory=VideoProcessor, async_mode=True)
