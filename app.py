import cv2
import numpy as np
import pygame
import streamlit as st

# Cài đặt âm thanh
pygame.mixer.init()
alarm_sound = r"D:\AI\Project\police.wav"

# Đọc các file cấu hình và class
config_file = r"D:\AI\Project\yolov3.cfg"
weights_file = r"D:\AI\Project\yolov3.weights"
classes_file = r"D:\AI\Project\yolov3.txt"

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
object_name = st.sidebar.text_input('Enter Object Name', 'cell phone')
frame_limit = st.sidebar.slider('Set Frame Limit for Alarm', 1, 10, 3)
start_button = st.button("Start")
stop_button = st.button("Stop")

# Trạng thái camera và chạy chương trình
if "cap" not in st.session_state:
    st.session_state.cap = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "object_not_found_counter" not in st.session_state:
    st.session_state.object_not_found_counter = 0

# Khi nhấn nút Start
if start_button:
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)  # Mở lại camera
    st.session_state.is_running = True

# Khi nhấn nút Stop
if stop_button:
    st.session_state.is_running = False
    if st.session_state.cap and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        st.session_state.cap = None  # Đặt lại trạng thái camera
    cv2.destroyAllWindows()

# Vòng lặp xử lý camera
if st.session_state.is_running:
    while st.session_state.is_running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.write("Không thể đọc camera. Vui lòng kiểm tra thiết bị!")
            break

        height, width, channels = frame.shape
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
                if confidence > 0.5 and classes[class_id] == object_name:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        object_found = False

        if len(indices) > 0:
            object_found = True
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                color = COLORS[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if not object_found:
            st.session_state.object_not_found_counter += 1
        else:
            st.session_state.object_not_found_counter = 0

        if st.session_state.object_not_found_counter >= frame_limit:
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play()

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            st.session_state.is_running = False

    # Dừng camera sau khi kết thúc
    if st.session_state.cap and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        st.session_state.cap = None
    cv2.destroyAllWindows()
