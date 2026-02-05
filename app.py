import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# Streamlit page setup
st.set_page_config(page_title="Skin Health Analysis", layout="centered")
st.title("🧴 Skin Health Analysis (Live Face Mesh)")


# Initialize MediaPipe FaceMesh using official API
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

        return img


st.subheader("Live Camera")

webrtc_streamer(
    key="skin-health",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Allow camera access to start real-time skin analysis.")
