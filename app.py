import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import os
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


MODEL_PATH = "Alphabet_knn_model.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "alphabet_knn_model.pkl"

model = joblib.load(MODEL_PATH)


st.set_page_config(page_title="Air Draw Alphabet Recognition", layout="wide")
st.title("✋ :rainbow[Air Draw Alphabet Recognition using MediaPipe]")
st.write(":green[Draw a large alphabet using your index finger, then click Predict]")


if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "processed_img" not in st.session_state:
    st.session_state.processed_img = None


def predict_alphabet(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Detect black drawing from white canvas
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return "No drawing", None, None

    # Crop only drawn area
    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y:y+h, x:x+w]

    # Add padding
    cropped = cv2.copyMakeBorder(
        cropped,
        20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # Make it square
    height, width = cropped.shape
    size = max(height, width)

    square = np.ones((size, size), dtype=np.uint8) * 255

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    square[y_offset:y_offset+height, x_offset:x_offset+width] = cropped

    # Resize to 32x32
    resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

    # Invert: training data is white letter on black background
    inverted = 255 - resized

    # Convert grayscale to 3 channels because model trained on 32x32x3
    final_img = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    # Flatten and normalize
    input_data = final_img.flatten() / 255.0
    input_data = input_data.reshape(1, -1)

    pred = model.predict(input_data)[0]

    conf = None
    if hasattr(model, "predict_proba"):
        conf = np.max(model.predict_proba(input_data)) * 100

    return pred, conf, final_img


class HandDrawingProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.prev_x = None
        self.prev_y = None
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]

                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

                with self.lock:
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(
                            self.canvas,
                            (self.prev_x, self.prev_y),
                            (x, y),
                            (0, 0, 0),
                            18
                        )

                self.prev_x = x
                self.prev_y = y

                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        else:
            self.prev_x = None
            self.prev_y = None

        with self.lock:
            small_canvas = cv2.resize(self.canvas, (220, 160))

        img[10:170, w-230:w-10] = small_canvas

        return frame.from_ndarray(img, format="bgr24")


col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="air-draw",
        video_processor_factory=HandDrawingProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader(":blue[Canvas Preview]")

    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            canvas_display = webrtc_ctx.video_processor.canvas.copy()

        st.image(canvas_display, channels="BGR", caption="Your Drawing")

        if st.button("🔮 Predict"):
            label, conf, processed_img = predict_alphabet(canvas_display)

            st.session_state.prediction = label
            st.session_state.confidence = conf
            st.session_state.processed_img = processed_img
            st.rerun()

        if st.button("🧹 Clear Canvas"):
            with webrtc_ctx.video_processor.lock:
                webrtc_ctx.video_processor.canvas = np.ones(
                    (480, 640, 3), dtype=np.uint8
                ) * 255

            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.processed_img = None
            st.rerun()

        if st.session_state.prediction is not None:
            if st.session_state.confidence is not None:
                st.success(
                    f"Predicted Symbol: {st.session_state.prediction} "
                    f"| Confidence: {st.session_state.confidence:.1f}%"
                )
            else:
                st.success(f"Predicted Symbol: {st.session_state.prediction}")

        if st.session_state.processed_img is not None:
            st.image(
                st.session_state.processed_img,
                channels="BGR",
                caption="Image Sent to Model",
                width=150
            )

    else:
        st.warning("Waiting for webcam to start...")

st.info("Draw clearly and large. Your training data is white letters on black background, so this app converts your drawing to that format before prediction.")