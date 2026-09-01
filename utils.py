"""
Shared utilities for the SmartVision AI multi-page app.
All pages import from this file so model loading logic (and caching) is
defined once, not duplicated across 5 separate page scripts.

IMPORTANT: each model below uses whichever version actually succeeded during
training - VGG16 uses v1 (its v2 retraining attempt crashed and was lost),
while ResNet50, MobileNetV2, and EfficientNetB0 use their v4/v2 versions
(which completed successfully with substantially better accuracy). Each
architecture below must exactly match what was actually trained and saved,
or loading the weights will fail with a shape mismatch.
"""
import json
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
}

# Maps each model to the actual weight file that succeeded in training.
MODEL_WEIGHT_FILES = {
    "vgg16": "models/vgg16_best.pth",                        # v1 - v2 attempt crashed, using original
    "resnet50": "models/resnet50_v4_best.pth",                # v4 - full unfreeze, best result (76.91%)
    "mobilenetv2": "models/mobilenetv2_v2_best.pth",          # v2 - expanded dataset (71.33%)
    "efficientnetb0": "models/efficientnetb0_v2_best.pth",    # v2 - expanded dataset (71.96%)
}


@st.cache_resource
def load_classes():
    with open("data/classes.json") as f:
        class_to_id = json.load(f)
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    return class_names


def build_model_architecture(model_key, num_classes):
    """
    Rebuilds the EXACT architecture that was actually trained and saved for
    each model - these differ from each other because different notebook
    versions succeeded for different models (see MODEL_WEIGHT_FILES above).
    """
    if model_key == "vgg16":
        # v1 architecture: single Linear layer replacing the final layer only
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    elif model_key == "resnet50":
        # v4 architecture: custom Sequential head (Linear->ReLU->Dropout->Linear)
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    elif model_key == "mobilenetv2":
        # v2 architecture: compact custom head with two dropout layers
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    elif model_key == "efficientnetb0":
        # v2 architecture: custom head with batch normalization
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return model


@st.cache_resource
def load_single_classifier(model_key, num_classes):
    model = build_model_architecture(model_key, num_classes)
    weights_path = MODEL_WEIGHT_FILES[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_all_classifiers(num_classes):
    """
    Loads all 4 CNN models at once, for the side-by-side comparison page.
    NOTE: this uses noticeably more memory than loading just one model -
    on a free-tier cloud deployment with limited RAM, this is the most
    memory-intensive part of the app.
    """
    models_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in MODEL_DISPLAY_NAMES:
        model = build_model_architecture(key, num_classes)
        weights_path = MODEL_WEIGHT_FILES[key]
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        models_dict[key] = model
    return models_dict, device


@st.cache_resource
def load_detection_model():
    from ultralytics import YOLO
    return YOLO("models/yolov8_best.pt")


@st.cache_resource
def load_all_summaries():
    """
    Loads the best available result summary per model (checking v4, v3, v2,
    then v1 in that order) - matches the same auto-selection logic used in
    the comparison notebook, so the app and notebook always agree on which
    result is being shown.
    """
    summaries = {}
    for base_key in MODEL_DISPLAY_NAMES:
        for suffix in ["_v4", "_v3", "_v2", ""]:
            path = f"results/{base_key}{suffix}_summary.json"
            if os.path.exists(path):
                with open(path) as f:
                    summaries[base_key] = json.load(f)
                break
    try:
        with open("results/yolov8_summary.json") as f:
            summaries["yolov8"] = json.load(f)
    except FileNotFoundError:
        pass
    return summaries
