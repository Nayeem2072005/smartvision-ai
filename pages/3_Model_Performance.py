"""
SmartVision AI - Model Performance Page
Comparison dashboard as specified in the brief ("Model comparison dashboard,
accuracy metrics visualization, inference speed comparison, confusion
matrices, class-wise performance breakdown").
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import load_all_summaries, MODEL_DISPLAY_NAMES

st.set_page_config(page_title="Model Performance", page_icon=":bar_chart:", layout="wide")
st.title("Model Performance Dashboard")

summaries = load_all_summaries()

if not summaries:
    st.error("No result files found. Run the training notebooks first "
             "(they save results to the results/ folder).")
    st.stop()

st.markdown("### Classification Models Comparison")

# summaries already contains the best available version per model
# (utils.load_all_summaries picks v4/v3/v2/v1 automatically)
classification_keys = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
available_keys = [k for k in classification_keys if k in summaries]

if available_keys:
    names = [summaries[k]["model_name"] for k in available_keys]
    accs = [summaries[k]["test_accuracy"] * 100 for k in available_keys]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)])
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Test Accuracy by Model")
        ax.set_ylim(0, 100)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.1f}%", ha="center")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)

    with col2:
        params = [summaries[k]["total_params"] / 1e6 for k in available_keys]
        times = [summaries[k]["training_time_minutes"] for k in available_keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, params, color="steelblue")
        ax.set_ylabel("Parameters (millions)")
        ax.set_title("Model Size Comparison")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)

    st.markdown("#### Summary table")
    table_data = []
    for k in available_keys:
        r = summaries[k]
        table_data.append({
            "Model": r["model_name"],
            "Test Accuracy": f"{r['test_accuracy']*100:.2f}%",
            "Parameters (M)": f"{r['total_params']/1e6:.1f}",
            "Training Time (min)": f"{r['training_time_minutes']:.1f}",
        })
    st.table(table_data)

    best_idx = int(np.argmax(accs))
    st.success(f"Best performing model: **{names[best_idx]}** ({accs[best_idx]:.2f}% test accuracy)")

st.markdown("---")
st.markdown("### YOLOv8 Detection Performance")

if "yolov8" in summaries:
    y = summaries["yolov8"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("mAP50", f"{y['map50']*100:.1f}%")
    col2.metric("mAP50-95", f"{y['map50_95']*100:.1f}%")
    col3.metric("Precision", f"{y['precision']*100:.1f}%")
    col4.metric("Recall", f"{y['recall']*100:.1f}%")

    with st.expander("Per-class detection performance"):
        sorted_classes = sorted(y["per_class_map"].items(), key=lambda x: -x[1])
        for cls, score in sorted_classes:
            st.progress(score, text=f"{cls}: {score*100:.1f}% mAP")
else:
    st.info("YOLOv8 results not found.")

st.markdown("---")
st.markdown("### Confusion Matrices")
st.markdown("Diagonal = correct predictions. Off-diagonal bright spots show which classes get confused with each other.")

cm_cols = st.columns(2)
def find_confusion_matrix_path(base_key):
    for suffix in ["_v4", "_v3", "_v2", ""]:
        path = f"results/{base_key}{suffix}_confusion_matrix.png"
        if os.path.exists(path):
            return path
    return None

cm_files = {
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}
for idx, (key, name) in enumerate(cm_files.items()):
    path = find_confusion_matrix_path(key)
    if path:
        with cm_cols[idx % 2]:
            st.image(path, caption=f"{name} Confusion Matrix", use_container_width=True)
