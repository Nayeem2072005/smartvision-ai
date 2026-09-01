"""
SmartVision AI - About Page
Project documentation as specified in the brief ("Project documentation,
dataset information, model architectures used, technical stack details,
developer information").
"""
import streamlit as st

st.set_page_config(page_title="About", page_icon=":information_source:", layout="wide")
st.title("About SmartVision AI")

st.markdown("""
### Project Overview

**SmartVision AI** is a computer vision capstone project combining transfer
learning-based image classification with YOLOv8 object detection, built as
part of the GUVI/HCL Data Science & AI program.

### Dataset

- **Source:** COCO (Common Objects in Context) 2017, via the
  `detection-datasets/coco` dataset on Hugging Face, streamed rather than
  fully downloaded
- **Scope:** 25 curated classes spanning vehicles, animals, furniture,
  electronics, and everyday items
- **Classification dataset:** cropped single-object images extracted from
  each object's bounding box (not full multi-object scenes) - matching
  standard image classification methodology
- **Detection dataset:** full images with all bounding boxes, in YOLO format

### Model Architectures

| Model | Type | Approach |
|---|---|---|
| VGG16 | CNN classifier | Frozen backbone, trained final layer only |
| ResNet50 | CNN classifier | Frozen backbone (v1) and deeper fine-tuning with unfrozen layer4 (v2) |
| MobileNetV2 | CNN classifier | Frozen backbone, lightweight/mobile-optimized |
| EfficientNetB0 | CNN classifier | Frozen backbone, compound-scaled architecture |
| YOLOv8n | Object detector | Fine-tuned from COCO-pretrained weights |

### Technical Stack

- **Framework:** PyTorch (torch, torchvision)
- **Detection:** Ultralytics YOLOv8
- **Data handling:** Hugging Face `datasets` (streaming), Pillow
- **Evaluation:** scikit-learn (classification metrics, confusion matrices)
- **Visualization:** Matplotlib, Seaborn
- **Web app:** Streamlit (multi-page)
- **Training hardware:** NVIDIA RTX 3070 Ti (local GPU)

### Known Limitations (documented honestly)

- Classification accuracy (~65% best model) is below the dataset's expected
  benchmark range, primarily due to visual overlap between related classes
  (e.g. truck/car/bus, handbag/backpack) and a relatively small number of
  training images per class for a 25-class problem
- YOLOv8 detection mAP is lower than large-scale COCO benchmarks, reflecting
  the smaller fine-tuning dataset used here (~1,750 training images) versus
  COCO's full 118K-image training set
- These are transparently reported rather than hidden, since understanding
  *why* a result falls short is more valuable than an unexplained high number

### Developer

Built by [Your Name] as a capstone project for the GUVI/HCL Data Science &
AI program, domain: Computer Vision & Artificial Intelligence.
""")
