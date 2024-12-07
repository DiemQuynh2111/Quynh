import cv2
import numpy as np
import streamlit as st
import time

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
def detect_objects_from_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Sử dụng API V4L2 cho Linux

    if not cap.isOpened():
        st.error("Không thể mở webcam. Kiểm tra quyền truy cập hoặc thử chỉ số camera khác.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể nhận khung hình từ webcam.")
            break

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

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                color = COLORS[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Hiển thị khung hình
        st.image(frame, channels="BGR", use_column_width=True)

        # Giới hạn tốc độ khung hình
        time.sleep(0.03)

    cap.release()

# Streamlit UI
st.title("Phát Hiện Đối Tượng với YOLO")

# Button để bắt đầu nhận diện đối tượng
if st.button("Bắt Đầu Phát Hiện"):
    detect_objects_from_webcam()

