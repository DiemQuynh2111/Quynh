import cv2
import numpy as np
import gradio as gr
import os
import gdown
import logging

# Cấu hình logging để theo dõi lỗi
logging.basicConfig(level=logging.DEBUG)

# Kiểm tra và tải tệp yolov3.weights từ Google Drive nếu chưa tồn tại
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    print("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Kiểm tra và tải các tệp cấu hình
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    print("Tệp cấu hình hoặc tệp classes không tồn tại. Vui lòng kiểm tra lại!")
    exit()

# Đọc các lớp từ tệp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Tạo màu sắc ngẫu nhiên cho các lớp đối tượng
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    exit()

# Hàm lấy các lớp đầu ra từ YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Hàm xử lý YOLO
def detect_objects(frame, object_names, frame_limit, object_counts_input):
    if frame is None:
        return frame  # Bỏ qua nếu khung hình là None
    
    height, width, _ = frame.shape
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

    # Áp dụng Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        color = COLORS[class_ids[i]]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        detected_objects[classes[class_ids[i]].lower()] += 1

    return frame

# Hàm để xử lý video trong Gradio
def video_processing(frame, object_names, frame_limit, object_counts_input):
    if frame is None:
        return frame  # Trả về None nếu frame là None

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Đổi từ BGR sang RGB
        processed_frame = detect_objects(frame, object_names, frame_limit, object_counts_input)
        return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Đổi lại từ RGB sang BGR
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý video: {e}")
        return frame

# Tạo giao diện Gradio
def create_interface():
    # Stream video từ webcam
    video_input = gr.inputs.Video(source="webcam", type="numpy")  # Nhận input video từ webcam
    video_output = gr.outputs.Video(type="numpy")  # Đầu ra video đã xử lý

    # Các tham số đầu vào từ người dùng
    object_names_input = gr.inputs.Textbox(default="cell phone,laptop,umbrella", label="Enter Object Names (comma separated)")
    frame_limit = gr.inputs.Slider(minimum=1, maximum=10, default=3, label="Set Frame Limit for Alarm")

    # Nhập số lượng vật thể cần giám sát
    object_counts_input = gr.inputs.Textbox(default="cell phone:1,laptop:1,umbrella:1", label="Enter number of objects to monitor (comma separated)")

    # Tạo giao diện Gradio
    iface = gr.Interface(fn=video_processing, 
                         inputs=[video_input, object_names_input, frame_limit, object_counts_input], 
                         outputs=video_output,
                         live=True)

    iface.launch()

# Khởi chạy giao diện
if __name__ == "__main__":
    create_interface()
