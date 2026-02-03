import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("WebRTC Camera Test")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame

webrtc_streamer(
    key="test",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
