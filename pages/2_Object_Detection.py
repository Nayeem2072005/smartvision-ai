"""
SmartVision AI - Object Detection Page
Upload an image, run YOLOv8, adjustable confidence threshold - as specified
in the brief ("YOLO detection with bounding boxes, display all detected
objects with labels, show detection confidence scores, option to adjust
confidence threshold").
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter

from utils import load_detection_model

st.set_page_config(page_title="Object Detection", page_icon=":mag:", layout="wide")
st.title("Object Detection (YOLOv8)")
st.markdown("Upload an image to detect and locate all objects in it.")

try:
    yolo_model = load_detection_model()
except FileNotFoundError as e:
    st.error(f"Could not load detection model: {e}")
    st.error("Make sure models/yolov8_best.pt is present.")
    st.stop()

confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
    help="Only show detections the model is at least this confident about. "
         "Lower values show more detections (including possibly wrong ones); "
         "higher values show fewer but more reliable detections."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting objects..."):
        # Pass the PIL image itself: Ultralytics converts it to the BGR order it expects.
        # (A plain NumPy array is assumed to be BGR, which swapped red and blue before.)
        results = yolo_model.predict(image, conf=confidence_threshold, verbose=False)[0]
        annotated = results.plot()[:, :, ::-1]  # BGR -> RGB for display

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader(f"Detected Objects (threshold: {confidence_threshold})")
        st.image(annotated, use_container_width=True)

    num_detections = len(results.boxes)
    st.markdown("---")
    st.subheader(f"{num_detections} object(s) detected")

    if num_detections > 0:
        detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
        confidences = [float(c) for c in results.boxes.conf]
        counts = Counter(detected_classes)

        cols = st.columns(min(len(counts), 5))
        for idx, (name, count) in enumerate(counts.items()):
            with cols[idx % len(cols)]:
                st.metric(label=name, value=count)

        with st.expander("See individual detection confidence scores"):
            for name, conf in zip(detected_classes, confidences):
                st.progress(conf, text=f"{name}: {conf*100:.1f}% confidence")
    else:
        st.info("No objects detected at this confidence threshold. Try lowering the slider above.")
else:
    st.info("Upload an image above to get started.")
