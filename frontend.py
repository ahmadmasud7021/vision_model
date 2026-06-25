import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


DEFAULT_API_URL = "http://localhost:8000"


st.set_page_config(page_title="VisionScene AI", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }

        .hero {
            padding: 2rem 0 1.25rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.16);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 2.45rem;
            line-height: 1.1;
            margin: 0 0 0.5rem;
        }

        .hero p {
            color: #586069;
            font-size: 1.05rem;
            margin: 0;
            max-width: 720px;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.4rem 0 1rem;
        }

        .metric-box {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 0.85rem;
            background: #ffffff;
        }

        .metric-label {
            color: #6a737d;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: #24292f;
            font-size: 1.35rem;
            font-weight: 700;
        }

        .caption-box {
            border-left: 4px solid #2f80ed;
            background: #f6f8fa;
            padding: 1rem 1.1rem;
            border-radius: 6px;
            color: #24292f;
            font-size: 1.05rem;
            line-height: 1.5;
        }

        .small-note {
            color: #6a737d;
            font-size: 0.88rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 2.8rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>VisionScene AI</h1>
        <p>Upload an image and inspect the generated caption, detected objects, counts, and OpenCV bounding-box preview.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).rstrip("/")
    st.caption("Use localhost while developing. Replace this with your deployed API URL on Render.")

    st.header("Upload")
    uploaded_file = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    left_column, right_column = st.columns([0.95, 1.05], gap="large")

    with left_column:
        st.subheader("Original")
        st.image(uploaded_file, use_container_width=True)
        st.markdown(
            f'<p class="small-note">{uploaded_file.name} - {uploaded_file.size / 1024:.1f} KB</p>',
            unsafe_allow_html=True,
        )

    with right_column:
        st.subheader("Analysis")

        if st.button("Analyze image", type="primary"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

            with st.spinner("Analyzing image..."):
                try:
                    response = requests.post(f"{api_url}/analyze", files=files, timeout=120)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    st.error(f"Analysis failed: {exc}")
                else:
                    result = response.json()
                    detections = result.get("detections", [])
                    counts = result.get("counts", {})

                    st.markdown(
                        f"""
                        <div class="metric-strip">
                            <div class="metric-box">
                                <div class="metric-label">Objects Detected</div>
                                <div class="metric-value">{len(detections)}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Unique Classes</div>
                                <div class="metric-value">{len(counts)}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Image Size</div>
                                <div class="metric-value">{uploaded_file.size / 1024:.0f} KB</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Caption")
                    st.markdown(
                        f'<div class="caption-box">{result["caption"]}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Object counts")
                    if counts:
                        count_rows = [
                            {"Object": key, "Count": value}
                            for key, value in sorted(counts.items())
                        ]
                        st.dataframe(count_rows, use_container_width=True, hide_index=True)
                    else:
                        st.info("No objects detected above the confidence threshold.")

                    st.markdown("#### Annotated image")
                    annotated_bytes = base64.b64decode(result["annotated_image_base64"])
                    annotated_image = Image.open(BytesIO(annotated_bytes))
                    st.image(annotated_image, use_container_width=True)

                    if detections:
                        st.markdown("#### Detections")
                        st.dataframe(detections, use_container_width=True, hide_index=True)
else:
    st.info("Upload an image from the sidebar to begin.")
