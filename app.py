import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Face Analysis App")
st.title("🧑 Face Mesh & Skin Analysis")

captured_image = st.empty()

class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="face-mesh",
    video_processor_factory=FaceMeshProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    if st.button("📸 Capture Face Image"):
        img = ctx.video_processor.latest_frame
        if img is not None:
            cv2.imwrite("captured_face.jpg", img)
            st.success("Image Captured!")
            captured_image.image(img, channels="BGR")


def analyze_skin(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- Pigmentation ----------
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    pigmentation_map = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    pigmentation_score = np.sum(pigmentation_map) / pigmentation_map.size

    pigmentation = "Detected" if pigmentation_score > 15 else "Not Significant"

    # ---------- Wrinkles ----------
    edges = cv2.Canny(gray, 80, 150)
    wrinkle_score = np.sum(edges) / edges.size
    wrinkles = "Detected" if wrinkle_score > 10 else "Not Significant"

    return pigmentation, wrinkles



if st.button("🧾 Generate Skin Report"):
    try:
        pigmentation, wrinkles = analyze_skin("captured_face.jpg")

        st.subheader("📊 Skin Analysis Report")
        st.write(f"🟤 **Pigmentation:** {pigmentation}")
        st.write(f"🧓 **Wrinkles:** {wrinkles}")

    except:
        st.error("Please capture an image first.")


import plotly.graph_objects as go

def generate_3d_face_mesh(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        x = [lm.x for lm in landmarks]
        y = [lm.y for lm in landmarks]
        z = [lm.z for lm in landmarks]

        return x, y, z


def plot_3d_face(x, y, z):
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            )
        )]
    )

    fig.update_layout(
        title="🧊 3D Face Model",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig



if st.button("🧊 Generate 3D Face Model"):
    data = generate_3d_face_mesh("captured_face.jpg")

    if data:
        x, y, z = data
        fig = plot_3d_face(x, y, z)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Face not detected. Please capture again.")
