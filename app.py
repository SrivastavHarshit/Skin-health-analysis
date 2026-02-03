import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(page_title="Skin Health Analysis", layout="centered")

st.title("🧑‍⚕️ Skin Health Analysis System")
st.info("Upload a clear face image to generate report and 3D visualization")

# ------------------ Image Upload ------------------
uploaded = st.file_uploader(
    "📸 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    with open("captured_face.jpg", "wb") as f:
        f.write(uploaded.read())

    img = cv2.imread("captured_face.jpg")
    st.image(img, channels="BGR", caption="Uploaded Image")

    # ------------------ Skin Analysis ------------------
    def analyze_skin_detailed(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_brightness = np.mean(gray)
        lighting = "Good" if 90 < mean_brightness < 180 else "Poor"

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        pigmentation_mask = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        pigmentation_pct = (np.sum(pigmentation_mask > 0) / pigmentation_mask.size) * 100
        pigmentation_level = (
            "Low" if pigmentation_pct < 8 else
            "Moderate" if pigmentation_pct < 15 else
            "High"
        )

        edges = cv2.Canny(gray, 80, 150)
        wrinkle_pct = (np.sum(edges > 0) / edges.size) * 100
        wrinkle_level = (
            "Low" if wrinkle_pct < 3 else
            "Moderate" if wrinkle_pct < 7 else
            "High"
        )

        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        pore_level = (
            "Low" if texture_var < 80 else
            "Moderate" if texture_var < 150 else
            "High"
        )

        return {
            "lighting": lighting,
            "pigmentation_pct": round(pigmentation_pct, 2),
            "pigmentation_level": pigmentation_level,
            "wrinkle_pct": round(wrinkle_pct, 2),
            "wrinkle_level": wrinkle_level,
            "pore_level": pore_level,
            "texture_score": round(texture_var, 2)
        }

    if st.button("🧾 Generate Detailed Skin Report"):
        report = analyze_skin_detailed(img)

        st.subheader("📊 Detailed Skin Report")

        st.markdown("### 🌤 Image Quality")
        st.write(f"Lighting: **{report['lighting']}**")

        st.markdown("### 🟤 Pigmentation")
        st.write(f"Coverage: **{report['pigmentation_pct']}%**")
        st.write(f"Severity: **{report['pigmentation_level']}**")

        st.markdown("### 🧓 Wrinkles")
        st.write(f"Density: **{report['wrinkle_pct']}%**")
        st.write(f"Severity: **{report['wrinkle_level']}**")

        st.markdown("### 🧴 Skin Texture")
        st.write(f"Pore Visibility: **{report['pore_level']}**")
        st.write(f"Texture Score: **{report['texture_score']}**")

        st.warning(
            "⚠️ This analysis is based on image processing and is for "
            "educational purposes only. Not a medical diagnosis."
        )

    # ------------------ 3D Face + Heatmap ------------------
    def generate_3d_face_heatmap(image):
        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 80, 150)
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        heatmap = cv2.normalize(edges + texture, None, 0, 1, cv2.NORM_MINMAX)

        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None

            x, y, z, intensity = [], [], [], []
            for lm in results.multi_face_landmarks[0].landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                x.append(lm.x)
                y.append(lm.y)
                z.append(lm.z)
                intensity.append(heatmap[py, px] if 0 <= px < w and 0 <= py < h else 0)

            return x, y, z, intensity

    if st.button("🔥 Generate 3D Face + Heatmap"):
        data = generate_3d_face_heatmap(img)

        if data:
            x, y, z, intensity = data
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=intensity,
                        colorscale="Jet",
                        opacity=0.85,
                        colorbar=dict(title="Skin Issue Intensity")
                    )
                )]
            )
            fig.update_layout(
                title="🧊 3D Face Model with Heatmap",
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Face not detected. Please upload a clear image.")
