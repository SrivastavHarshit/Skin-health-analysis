import streamlit as st
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.set_page_config(page_title="Skin Health Analysis", layout="centered")
st.title("🧴 Skin Health Analysis (Live Face Detection)")


# Load MediaPipe Face Detector (Tasks API)
base_options = python.BaseOptions(model_asset_path=None)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img


webrtc_streamer(
    key="skin-health",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("Allow camera access to start live detection.")
