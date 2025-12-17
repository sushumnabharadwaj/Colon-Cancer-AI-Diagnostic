import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import xgboost as xgb
import pandas as pd
import numpy as np

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Colon Cancer AI Diagnostic", layout="wide")
st.title("☁️ Cloud-AI: Multi-Modal Colon Cancer Diagnostic System")

# --- 2. THE IMAGE AI (CNN) ---
@st.cache_resource
def load_cnn_model():
    # We use a pre-trained ResNet-18 (Standard for medical imaging)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2) # 2 classes: Cancer vs Benign
    model.eval()
    return model

def predict_image(img, model):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
    return prob[1].item() # Return probability of Cancer

# --- 3. THE GENOMIC AI (XGBoost) ---
def genomic_analysis(gene_name, allele_freq):
    # Simulation of XGBoost pathogenicity scoring
    # In a real app, this would be: model.predict(features)
    risk_score = 0.85 if allele_freq < 0.0001 else 0.15
    if gene_name in ["TP53", "KRAS", "APC"]:
        risk_score += 0.1
    return min(risk_score, 1.0)

# --- 4. THE USER INTERFACE (UI) ---
col1, col2 = st.columns(2)

with col1:
    st.header("🔬 Histopathology Analysis")
    uploaded_file = st.file_uploader("Upload Biopsy Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Biopsy", use_column_width=True)
        cnn_model = load_cnn_model()
        img_risk = predict_image(image, cnn_model)
        st.write(f"**Image-based Malignancy Probability:** {img_risk:.2%}")

with col2:
    st.header("🧬 Genomic Analysis")
    gene = st.selectbox("Select Target Gene", ["APC", "TP53", "KRAS", "BRAF", "MSH2"])
    freq = st.number_input("Allele Frequency (from gnomAD)", format="%.6f", value=0.000050)
    
    if st.button("Run Genomic Scoring"):
        gen_risk = genomic_analysis(gene, freq)
        st.session_state['gen_risk'] = gen_risk
        st.write(f"**Variant Pathogenicity Score:** {gen_risk:.2%}")

# --- 5. FUSION & CLOUD REPORTING ---
st.divider()
if uploaded_file and 'gen_risk' in st.session_state:
    st.header("📊 Final Cloud-Fused Diagnostic Result")
    # Simple Fusion logic: 60% Image weight + 40% Genetic weight
    final_score = (img_risk * 0.6) + (st.session_state['gen_risk'] * 0.4)
    
    if final_score > 0.7:
        st.error(f"HIGH RISK DETECTED: {final_score:.2%}")
        st.write("**Recommendation:** Immediate Oncology referral and BRAF/MSI testing.")
    else:
        st.success(f"LOW RISK DETECTED: {final_score:.2%}")
        st.write("**Recommendation:** Follow-up screening in 6 months.")
