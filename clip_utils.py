import streamlit as st

from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model and processor
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Predict tags using CLIP
def get_clip_tags(image, labels=None):
    if labels is None:
        labels = ["electronics", "furniture", "books", "clothing", "toys", "accessories", "home", "beauty", "tools"]

    model, processor = load_clip_model()
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    top_probs, top_idxs = probs.topk(3)
    return [labels[i] for i in top_idxs[0]]
