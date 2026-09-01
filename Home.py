"""
SmartVision AI - Home Page
Run with: streamlit run Home.py
"""
import streamlit as st

st.set_page_config(page_title="SmartVision AI - Home", page_icon=":camera:", layout="wide")

st.title("SmartVision AI")
st.subheader("Intelligent Multi-Class Object Recognition System")

st.markdown("""
### Project Overview

SmartVision AI is a computer vision platform combining **transfer learning-based
image classification** with **YOLOv8 object detection**, trained on a curated
25-class subset of the COCO dataset.

### Key Features

- **4 CNN architectures compared**: VGG16, ResNet50, MobileNetV2, EfficientNetB0
- **Real-time object detection** with YOLOv8, showing all objects and their locations
- **Side-by-side model comparison** on accuracy, speed, and size
- **Interactive predictions** on your own uploaded images

### How to Use This App

Use the sidebar on the left to navigate between pages:

1. **Image Classification** - upload an image and see predictions from all 4 CNN models
2. **Object Detection** - upload an image and see YOLOv8 detect and locate every object
3. **Model Performance** - compare all 5 models' accuracy, speed, and training curves
4. **About** - full project documentation, dataset details, and technical stack

### Dataset

25 object classes spanning vehicles, animals, furniture, electronics, and everyday
items, drawn from the COCO (Common Objects in Context) dataset - a standard
benchmark for object detection and classification research.
""")

st.info("Start by picking a page from the sidebar on the left.")
