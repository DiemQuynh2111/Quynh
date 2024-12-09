import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import os
import gdown
import base64

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

# Hàm để tạo âm thanh cảnh báo
def play_sound():
    sound_path = "alert.wav"
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as sound_file:
            audio_bytes = sound_file.read()
        st.audio(audio_bytes, format="audio/wav")

# Xác định lớp xử lý video
class VideoTransformer(VideoTransformerBase):
    def __init__(self, object_names, fps):
        self.object_names = object_names
        self.prev_objects = {obj: True for obj in object_names}
        self.fps = fps
        self.time_lost = {}

    def transform(self, frame):
        try:
            frame = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
            detected_objects = detect_objects(frame, self.object_names, self.prev_objects)

            for obj, detected in detected_objects.items():
                if not detected and self.prev_objects[obj]:
                    current_time = st.session_state["frame_count"] / self.fps
                    self.time_lost[obj] = current_time
                    play_sound()  # Phát âm thanh cảnh báo
                    st.warning(f"{obj.upper()} bị mất lúc: {int(current_time // 60)}:{int(current_time % 60)}")
                
                self.prev_objects[obj] = detected

            st.session_state["frame_count"] += 1
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Lỗi trong quá trình xử lý video: {e}")
            return frame.to_ndarray(format="bgr24")

# Streamlit UI
st.title("Object Detection with Alarm for Missing Objects")

# Nhập đối tượng cần theo dõi
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
fps_input = st.sidebar.slider('Enter Video FPS', 1, 60, 30)

# Cấu hình WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
})

# Khởi tạo bộ đếm khung hình
if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VideoTransformer(object_names, fps_input),
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
