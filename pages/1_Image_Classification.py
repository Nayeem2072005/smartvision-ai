"""
SmartVision AI - Image Classification Page
Shows predictions from all 4 trained CNN models side by side, as specified
in the project brief ("display predictions from all 4 models, show top-5
predictions with confidence scores, model comparison side-by-side").
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from utils import load_classes, load_all_classifiers, MODEL_DISPLAY_NAMES, IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(page_title="Image Classification", page_icon=":camera:", layout="wide")
st.title("Image Classification - All 4 Models Compared")
st.markdown("Upload an image to see how each of the 4 trained CNN architectures classifies it.")

try:
    class_names = load_classes()
    models_dict, device = load_all_classifiers(len(class_names))
except FileNotFoundError as e:
    st.error(f"Could not load model files: {e}")
    st.error("Make sure all 4 model weight files are present in the 'models/' folder.")
    st.stop()

classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)
    st.markdown("---")
    st.subheader("Predictions from each model")

    input_tensor = classify_transform(image).unsqueeze(0).to(device)

    loaded_models = list(models_dict.keys())
    top1_results = {}
    if len(loaded_models) < 4:
        skipped = [MODEL_DISPLAY_NAMES[k] for k in MODEL_DISPLAY_NAMES if k not in loaded_models]
        st.caption(f"Note: {', '.join(skipped)} not loaded in this deployment "
                   f"(large weight files excluded from cloud deployment - see About page)")

    cols = st.columns(len(loaded_models))
    for idx, model_key in enumerate(loaded_models):
        model_name = MODEL_DISPLAY_NAMES[model_key]
        model = models_dict[model_key]
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)

        top1_results[model_name] = (class_names[top5_indices[0]], float(top5_probs[0]))

        with cols[idx]:
            st.markdown(f"**{model_name}**")
            st.markdown(f"Top: **{class_names[top5_indices[0]]}** ({top5_probs[0]*100:.1f}%)")
            for prob, cidx in zip(top5_probs, top5_indices):
                st.progress(float(prob), text=f"{class_names[cidx]}: {prob*100:.1f}%")

    st.markdown("---")
    st.subheader("Quick comparison")
    agreement = len(set(v[0] for v in top1_results.values()))
    if agreement == 1:
        st.success(f"All {len(loaded_models)} models agree: **{list(top1_results.values())[0][0]}**")
    else:
        st.warning(f"Models disagree - {agreement} different top predictions across the 4 models. "
                   "This can happen on images that are ambiguous or unlike the training data "
                   "(see the Model Performance page for each model's known accuracy).")
else:
    st.info("Upload an image above to get started.")
