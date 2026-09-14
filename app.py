import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import time

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 5.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 40
MODEL_PATH = os.path.join('models', 'rf_model.pkl')

st.set_page_config(page_title="VoiceGuard", page_icon="🔬", layout="centered")

# Classic, Clean, Academic/GitHub-like Minimalist Theme
st.markdown("""
<style>
/* Clean System Fonts */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
    background-color: #ffffff !important;
    color: #24292f !important;
}
.stApp {
    background-color: #ffffff !important;
}

/* Hide standard Streamlit header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Typography */
.app-title {
    font-size: 24px;
    font-weight: 600;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 8px;
    margin-bottom: 8px;
    color: #24292f;
}
.app-desc {
    font-size: 14px;
    color: #57606a;
    margin-bottom: 32px;
}
.section-heading {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    margin-top: 12px;
    color: #24292f;
}

/* Standard Alerts */
.alert-success {
    background-color: #dafbe1;
    color: #1a7f37;
    border: 1px solid rgba(26,127,55,0.15);
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 16px;
}
.alert-danger {
    background-color: #ffebe9;
    color: #cf222e;
    border: 1px solid rgba(207,34,46,0.15);
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* Data Table */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 16px;
    color: #24292f;
}
.data-table th, .data-table td {
    border: 1px solid #d0d7de;
    padding: 8px 12px;
    text-align: left;
    background-color: #ffffff;
    color: #24292f;
}
.data-table th {
    background-color: #f6f8fa;
    color: #57606a;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.markdown("<div class='app-title'>VoiceGuard Analysis Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='app-desc'>A machine learning utility for detecting synthetically generated audio using MFCC feature extraction and Random Forest classification.</div>", unsafe_allow_html=True)

st.markdown("<div class='section-heading'>1. Data Input</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Select audio file (.wav format)", type=["wav"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

if uploaded_file is not None:
    st.markdown("<hr style='border: none; border-top: 1px solid #d0d7de; margin: 32px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>2. Analysis Results</div>", unsafe_allow_html=True)
    
    with st.spinner("Processing audio features..."):
        try:
            start_time = time.time()
            
            # Load and Preprocess Audio
            y, sr = librosa.load(uploaded_file, sr=TARGET_SR)
            
            # Pad or truncate
            if len(y) > TARGET_SAMPLES:
                y = y[:TARGET_SAMPLES]
            elif len(y) < TARGET_SAMPLES:
                padding = TARGET_SAMPLES - len(y)
                y = np.pad(y, (0, padding), 'constant')
                
            # Extract Features
            mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            X_input = mfcc_mean.reshape(1, -1)
            
            # Predict
            model = load_model()
            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]
            confidence = probabilities[prediction] * 100
            
            process_time = time.time() - start_time
            
            # Results
            if prediction == 0:
                st.markdown(f"<div class='alert-success'>Authentic Audio (Confidence: {confidence:.2f}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-danger'>Synthetic Audio Detected (Confidence: {confidence:.2f}%)</div>", unsafe_allow_html=True)
                
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                # Waveform Plot (Standard Scientific look)
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(6, 3))
                
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                librosa.display.waveshow(y, sr=TARGET_SR, ax=ax, color='#0969da')
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#d0d7de')
                ax.spines['bottom'].set_color('#d0d7de')
                ax.tick_params(colors='#57606a', labelsize=9)
                ax.set_xlabel('Time (s)', color='#57606a', fontsize=10)
                ax.set_ylabel('Amplitude', color='#57606a', fontsize=10)
                
                st.pyplot(fig)
                
            with col2:
                # Diagnostics Table
                table_html = f"""
                <table class="data-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Sample Rate</td><td>{TARGET_SR} Hz</td></tr>
                    <tr><td>Duration</td><td>{TARGET_DURATION}s</td></tr>
                    <tr><td>MFCC Features</td><td>{N_MFCC}</td></tr>
                    <tr><td>Classifier</td><td>Random Forest</td></tr>
                    <tr><td>Process Time</td><td>{process_time:.3f}s</td></tr>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")
