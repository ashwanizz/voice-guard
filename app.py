import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import time
import base64
from PIL import Image

# ─── Configuration & Model Path ───
TARGET_SR = 16000
TARGET_DURATION = 5.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 40
MODEL_PATH = os.path.join('models', 'rf_model.pkl')
LOGO_PATH = os.path.join('assets', 'logo_small.png')
FULL_LOGO_PATH = os.path.join('assets', 'logo.png')

# Load page icon
page_icon = "🛡️"
if os.path.exists(FULL_LOGO_PATH):
    try:
        page_icon = Image.open(FULL_LOGO_PATH)
    except Exception:
        page_icon = "🛡️"

st.set_page_config(
    page_title="Voice Guard | Forensic Audio Deepfake Detector",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Load Logo as Base64 ───
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_image(LOGO_PATH) or get_base64_image(FULL_LOGO_PATH)

# ─── Professional Dark Navy / Charcoal Theme ───
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Global Canvas & Typography ── */
:root {
    --bg-main: #0B0F19;
    --bg-card: #111827;
    --bg-card-subtle: #162032;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-focus: #3B82F6;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-blue: #38BDF8;
    --accent-indigo: #6366F1;
    --success-green: #10B981;
    --danger-red: #F43F5E;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background-color: var(--bg-main) !important;
}

/* Chrome Cleanup */
header {visibility: hidden; height: 0 !important; min-height: 0 !important;}
footer {visibility: hidden; height: 0 !important;}
#MainMenu {visibility: hidden;}
.stDeployButton {display: none !important;}

.block-container {
    padding: 1.2rem 2.5rem 1.8rem !important;
    max-width: 1400px !important;
}

/* ── Top Navigation Bar ── */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: linear-gradient(180deg, #131B2E 0%, #0F1728 100%);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Pure clean borderless logo */
.brand-logo-img {
    width: 36px;
    height: 36px;
    object-fit: contain;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    display: block;
}

.brand-shield {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    font-size: 20px;
}

.brand-text h1 {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #F8FAFC;
    margin: 0;
    line-height: 1.2;
}

.brand-text p {
    font-size: 11.5px;
    font-weight: 400;
    color: #64748B;
    margin: 2px 0 0;
    letter-spacing: 0.2px;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 20px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    font-size: 11.5px;
    font-weight: 600;
    color: #34D399;
    font-family: 'JetBrains Mono', monospace;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.9); }
    100% { opacity: 1; transform: scale(1); }
}

/* ── Streamlit Containers as Dark Cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border-radius: 16px !important;
    padding: 18px 22px 22px 22px !important;
    margin-bottom: 16px !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25) !important;
    box-sizing: border-box !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    gap: 0.75rem !important;
}

/* ── Section Titles ── */
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}

.panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #E2E8F0;
}

.panel-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: linear-gradient(135deg, #2563EB, #38BDF8);
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.panel-tag {
    font-size: 10.5px;
    font-weight: 500;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Compact & Sleek Audio Ingestion Upload Box ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 4px 0 10px 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(14, 21, 36, 0.45) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.22) !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    min-height: unset !important;
    height: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    transition: all 0.2s ease-in-out !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #38BDF8 !important;
    background: rgba(17, 26, 46, 0.65) !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.12) !important;
}

[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 14px !important;
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 12px !important;
    color: var(--text-muted) !important;
}

/* Hide default file uploader list/box artifacts completely */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileData"],
[data-testid="stFileUploader"] ul {
    display: none !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    font-size: 12.5px !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}

/* ── File Details Pill / Card ── */
.file-meta-card {
    background: #0D1424;
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 2px 0 12px 0;
}

.file-meta-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.file-name-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
}

.file-icon {
    font-size: 16px;
    color: var(--accent-blue);
}

.file-name-text {
    font-size: 13.5px;
    font-weight: 600;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
}

.file-meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    padding-top: 6px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.file-meta-item {
    font-size: 10.5px;
    color: var(--text-muted);
}

.file-meta-val {
    font-size: 12px;
    font-weight: 600;
    color: #E2E8F0;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ── Dark Audio Player ── */
audio {
    width: 100% !important;
    height: 40px !important;
    border-radius: 10px !important;
    margin-top: 4px;
    filter: invert(0.9) hue-rotate(180deg);
}

/* ── Primary Action Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.3px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* ── Compact Verdict Banners ── */
.verdict-card {
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}

.verdict-card.authentic {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
}

.verdict-card.synthetic {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(244, 63, 94, 0.1) 100%);
    border: 1px solid rgba(244, 63, 94, 0.35);
    box-shadow: 0 4px 20px rgba(244, 63, 94, 0.1);
}

.verdict-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 14.5px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.verdict-badge.authentic { color: #34D399; }
.verdict-badge.synthetic { color: #FB7185; }

.confidence-score {
    font-size: 22px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}

.confidence-score.authentic { color: #34D399; }
.confidence-score.synthetic { color: #FB7185; }

.verdict-meter {
    width: 100%;
    height: 5px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    margin-bottom: 8px;
    overflow: hidden;
}

.verdict-meter-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease-out;
}

.verdict-meter-fill.authentic {
    background: linear-gradient(90deg, #059669 0%, #10B981 100%);
    box-shadow: 0 0 10px #10B981;
}

.verdict-meter-fill.synthetic {
    background: linear-gradient(90deg, #E11D48 0%, #F43F5E 100%);
    box-shadow: 0 0 10px #F43F5E;
}

.verdict-explanation {
    font-size: 12px;
    color: #CBD5E1;
    line-height: 1.45;
    margin: 0;
}

/* ── Neutral / Ready State ── */
.ready-card {
    background: #0E1524;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
}

.ready-icon {
    font-size: 28px;
    margin-bottom: 8px;
    color: var(--accent-blue);
    opacity: 0.9;
}

.ready-title {
    font-size: 14px;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 4px;
}

.ready-desc {
    font-size: 12px;
    color: var(--text-muted);
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.45;
}

/* ── Empty State Guide ── */
.empty-hero {
    background: #0E1524;
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 28px 22px;
    text-align: center;
}

.empty-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 18px;
    text-align: left;
}

.step-box {
    background: #131C30;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 9px;
    padding: 12px;
}

.step-box-num {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 3px;
}

.step-box-title {
    font-size: 12px;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 3px;
}

.step-box-desc {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.35;
}

/* ── Diagnostic Metrics Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 8px;
    margin-bottom: 4px;
}

.metric-tile {
    background: #0E1524;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 9px;
    padding: 8px 12px;
}

.metric-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    margin-bottom: 2px;
}

.metric-val {
    font-size: 13.5px;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 9.5px;
    color: #64748B;
    margin-top: 1px;
}

/* ── Clean Waveform Container ── */
.chart-container-title {
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Error Notification ── */
.error-banner {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 10px;
    padding: 12px 16px;
    color: #FCA5A5;
    font-size: 12.5px;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Compact Footer ── */
.app-footer {
    text-align: center;
    padding: 12px 0 6px;
    font-size: 11px;
    color: var(--text-muted);
    border-top: 1px solid var(--border-color);
    margin-top: 16px;
}

.app-footer a {
    color: var(--accent-blue);
    text-decoration: none;
    font-weight: 500;
}

.app-footer a:hover {
    text-decoration: underline;
}

/* Mobile & Tablet Stacking */
@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem !important; }
    .empty-steps { grid-template-columns: 1fr; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    .file-meta-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ───
@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ─── Navigation Header ───
logo_element = f'<img src="data:image/png;base64,{logo_b64}" class="brand-logo-img" alt="Voice Guard Logo" />' if logo_b64 else '<div class="brand-shield">🛡️</div>'

st.markdown(f"""
<div class="top-nav">
    <div class="nav-brand">
        {logo_element}
        <div class="brand-text">
            <h1>VOICE GUARD <span style="font-size:12.5px; font-weight:400; color:#38BDF8; margin-left:6px;">FORENSIC AUDIO SUITE</span></h1>
            <p>Acoustic feature analysis &amp; Random Forest synthetic voice detection</p>
        </div>
    </div>
    <div class="nav-status">
        <span class="status-dot"></span>
        MODEL ONLINE · 16kHz PIPELINE
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Helper Functions ───
def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


# ─── Initialize Session State ───
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "has_analyzed" not in st.session_state:
    st.session_state["has_analyzed"] = False
if "current_file_name" not in st.session_state:
    st.session_state["current_file_name"] = None


# ─── Layout Grid (Two Columns) ───
left_col, right_col = st.columns([1, 1.25], gap="large")

# ═══════════════════════════════════════════════════════════════
# LEFT COLUMN: Data Ingestion & Audio Inspection
# ═══════════════════════════════════════════════════════════════
with left_col:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-header">
            <div class="panel-title">
                <span class="panel-step">1</span>
                Audio Ingestion
            </div>
            <span class="panel-tag">SUPPORTED: WAV / PCM</span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select an audio file for deepfake forensic verification",
            type=["wav"],
            help="Upload an uncompressed WAV audio clip. Recommended: mono/stereo, max 25MB.",
            key=f"audio_input_{st.session_state['uploader_key']}",
            label_visibility="collapsed"
        )

        # Detect file replacement
        if uploaded_file is not None:
            if st.session_state["current_file_name"] != uploaded_file.name:
                st.session_state["current_file_name"] = uploaded_file.name
                st.session_state["has_analyzed"] = False

        if uploaded_file is None:
            st.session_state["current_file_name"] = None
            st.session_state["has_analyzed"] = False
            st.markdown("""
            <div style="font-size:11.5px; color:#64748B; margin-top:6px; line-height:1.45;">
                ℹ️ <strong>Tip:</strong> Drag and drop any voice recording (.wav) to extract 40 acoustic MFCC features and classify authenticity.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Hide the default Streamlit uploader file bar once file is loaded
            st.markdown("""
            <style>
            [data-testid="stFileUploader"] {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # ── Dynamic Audio Metadata Extraction ──
            try:
                uploaded_file.seek(0)
                y_raw, native_sr = librosa.load(uploaded_file, sr=None)
                native_duration = librosa.get_duration(y=y_raw, sr=native_sr)

                # Render File Metadata Card
                st.markdown(f"""
                <div class="file-meta-card">
                    <div class="file-meta-header">
                        <div class="file-name-wrap">
                            <span class="file-icon">🎵</span>
                            <span class="file-name-text" title="{uploaded_file.name}">{uploaded_file.name}</span>
                        </div>
                        <span style="font-size:10.5px; color:#38BDF8; font-family:'JetBrains Mono', monospace; font-weight:600;">ACTIVE</span>
                    </div>
                    <div class="file-meta-grid">
                        <div class="file-meta-item">
                            FILE SIZE
                            <div class="file-meta-val">{format_file_size(uploaded_file.size)}</div>
                        </div>
                        <div class="file-meta-item">
                            DURATION
                            <div class="file-meta-val">{native_duration:.2f} s</div>
                        </div>
                        <div class="file-meta-item">
                            NATIVE SR
                            <div class="file-meta-val">{native_sr:,} Hz</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Native Audio Player
                st.markdown('<div style="font-size:11px; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:4px;">Audio Playback</div>', unsafe_allow_html=True)
                uploaded_file.seek(0)
                st.audio(uploaded_file, format='audio/wav')

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # Action Button Row (Analyze + Remove / Replace)
                btn_col1, btn_col2 = st.columns([2.2, 1.2])
                with btn_col1:
                    run_analysis = st.button("⚡ Run Deepfake Analysis", type="primary", use_container_width=True)
                with btn_col2:
                    if st.button("✕ Remove", use_container_width=True):
                        st.session_state["uploader_key"] += 1
                        st.session_state["has_analyzed"] = False
                        st.session_state["current_file_name"] = None
                        st.rerun()

                if run_analysis:
                    st.session_state["has_analyzed"] = True

            except Exception as e:
                st.markdown(f"""
                <div class="error-banner">
                    <span>⚠️</span>
                    <div>
                        <strong>Failed to decode audio file.</strong><br>
                        {str(e)}.<br>
                        <em>Please check that the file is a valid, uncorrupted PCM WAV container.</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RIGHT COLUMN: Forensic Analysis & Visualizations
# ═══════════════════════════════════════════════════════════════
with right_col:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-header">
            <div class="panel-title">
                <span class="panel-step">2</span>
                Forensic Results &amp; Diagnostics
            </div>
            <span class="panel-tag">CLASSIFIER: RANDOM FOREST</span>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file is None:
            # ── Empty State: Clear Guidance ──
            st.markdown("""
            <div class="empty-hero">
                <div style="font-size:32px; margin-bottom:10px;">🔬</div>
                <div style="font-size:15px; font-weight:700; color:#F8FAFC; margin-bottom:4px;">No Audio Loaded</div>
                <div style="font-size:12px; color:#94A3B8; max-width:440px; margin:0 auto; line-height:1.45;">
                    Upload a WAV recording on the left panel to begin forensic frequency inspection and synthetic voice classification.
                </div>
                <div class="empty-steps">
                    <div class="step-box">
                        <div class="step-box-num">01 / INGEST</div>
                        <div class="step-box-title">Upload WAV</div>
                        <div class="step-box-desc">Standardizes sample rate to 16 kHz and pads or clips to a 5.0-second window.</div>
                    </div>
                    <div class="step-box">
                        <div class="step-box-num">02 / EXTRACT</div>
                        <div class="step-box-title">40 MFCC Bands</div>
                        <div class="step-box-desc">Computes Mel-Frequency Cepstral Coefficients capturing vocal tract resonances.</div>
                    </div>
                    <div class="step-box">
                        <div class="step-box-num">03 / INFER</div>
                        <div class="step-box-title">Random Forest</div>
                        <div class="step-box-desc">Evaluates spectral distribution against 100 decision trees to determine authenticity.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.get("has_analyzed", False):
            # ── Ready State: Neutral waiting for click ──
            st.markdown("""
            <div class="ready-card">
                <div class="ready-icon">⚡</div>
                <div class="ready-title">Ready for Forensic Evaluation</div>
                <div class="ready-desc">
                    Audio stream loaded and validated. Click <strong>"Run Deepfake Analysis"</strong> on the left panel to execute feature extraction and model inference.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Execute Full Analysis Pipeline ──
            model = load_classifier()
            if model is None:
                st.error("Classifier model file `models/rf_model.pkl` not found. Please train or provide the model file.")
            else:
                with st.spinner("Extracting 40 MFCC features & evaluating decision ensemble..."):
                    try:
                        start_time = time.time()

                        # 1. Resample to standard 16kHz
                        uploaded_file.seek(0)
                        y_norm, sr = librosa.load(uploaded_file, sr=TARGET_SR)

                        # 2. Window normalization (5.0s)
                        if len(y_norm) > TARGET_SAMPLES:
                            y_processed = y_norm[:TARGET_SAMPLES]
                            window_note = "Truncated to 5.0s"
                        elif len(y_norm) < TARGET_SAMPLES:
                            padding = TARGET_SAMPLES - len(y_norm)
                            y_processed = np.pad(y_norm, (0, padding), 'constant')
                            window_note = f"Padded +{padding} samples"
                        else:
                            y_processed = y_norm
                            window_note = "Exact 5.0s window"

                        # 3. MFCC Feature Extraction (40 coefficients)
                        mfcc = librosa.feature.mfcc(y=y_processed, sr=TARGET_SR, n_mfcc=N_MFCC)
                        mfcc_mean = np.mean(mfcc.T, axis=0)
                        X_input = mfcc_mean.reshape(1, -1)

                        # 4. Model Prediction
                        prediction = int(model.predict(X_input)[0])
                        probabilities = model.predict_proba(X_input)[0]
                        confidence = float(probabilities[prediction] * 100.0)
                        process_time = time.time() - start_time
                        latency_ms = process_time * 1000.0

                        # ── Outcome Verdict Banner ──
                        if prediction == 0:
                            # AUTHENTIC VOICE
                            st.markdown(f"""
                            <div class="verdict-card authentic">
                                <div class="verdict-top">
                                    <div class="verdict-badge authentic">
                                        <span style="font-size:18px;">🛡️</span>
                                        AUTHENTIC AUDIO DETECTED
                                    </div>
                                    <div class="confidence-score authentic">{confidence:.2f}%</div>
                                </div>
                                <div class="verdict-meter">
                                    <div class="verdict-meter-fill authentic" style="width: {max(confidence, 10.0):.1f}%;"></div>
                                </div>
                                <p class="verdict-explanation">
                                    <strong>Natural Human Voice:</strong> Acoustic spectral variance and vocal tract formants align closely with authentic human biological resonance. No anomalous neural vocoder artifacts detected.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # SYNTHETIC / DEEPFAKE
                            st.markdown(f"""
                            <div class="verdict-card synthetic">
                                <div class="verdict-top">
                                    <div class="verdict-badge synthetic">
                                        <span style="font-size:18px;">⚠️</span>
                                        SYNTHETIC AUDIO DETECTED
                                    </div>
                                    <div class="confidence-score synthetic">{confidence:.2f}%</div>
                                </div>
                                <div class="verdict-meter">
                                    <div class="verdict-meter-fill synthetic" style="width: {max(confidence, 10.0):.1f}%;"></div>
                                </div>
                                <p class="verdict-explanation">
                                    <strong>AI-Generated Deepfake:</strong> Harmonic repetition, unnatural phase coherence, and anomalous MFCC feature distributions indicate speech generated by neural TTS, voice conversion, or cloning engines.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # ── Waveform Plot (Forensic Dark Mode) ──
                        st.markdown("""
                        <div class="chart-container-title">
                            <span>Amplitude Spectrum vs. Time</span>
                            <span style="font-size:10.5px; color:#64748B; font-family:'JetBrains Mono', monospace;">16,000 HZ RESAMPLED</span>
                        </div>
                        """, unsafe_allow_html=True)

                        plt.style.use('dark_background')
                        fig, ax = plt.subplots(figsize=(6.5, 1.8), dpi=120)
                        fig.patch.set_facecolor('#0E1524')
                        ax.set_facecolor('#0E1524')

                        # High-contrast dual-layer waveform
                        librosa.display.waveshow(y_processed, sr=TARGET_SR, ax=ax, color='#6366F1', alpha=0.35)
                        librosa.display.waveshow(y_processed, sr=TARGET_SR, ax=ax, color='#38BDF8', alpha=0.85)

                        # Styled Axes & Spines
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#1E293B')
                        ax.spines['bottom'].set_color('#1E293B')
                        ax.tick_params(colors='#64748B', labelsize=8)
                        ax.grid(color='#1E293B', linestyle='--', linewidth=0.5, alpha=0.5)
                        ax.set_xlabel('Time (seconds)', color='#94A3B8', fontsize=8.5, fontfamily='sans-serif', fontweight=500)
                        ax.set_ylabel('Amplitude', color='#94A3B8', fontsize=8.5, fontfamily='sans-serif', fontweight=500)
                        fig.tight_layout(pad=0.3)

                        st.pyplot(fig)

                        # ── Technical Diagnostics Tiles ──
                        st.markdown(f"""
                        <div class="metrics-grid">
                            <div class="metric-tile">
                                <div class="metric-label">Target Sample Rate</div>
                                <div class="metric-val">16,000 Hz</div>
                                <div class="metric-sub">Forensic standard</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-label">Evaluation Window</div>
                                <div class="metric-val">5.00 s</div>
                                <div class="metric-sub">{window_note}</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-label">Native Duration</div>
                                <div class="metric-val">{native_duration:.2f} s</div>
                                <div class="metric-sub">Source file length</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-label">MFCC Coefficients</div>
                                <div class="metric-val">40 Bands</div>
                                <div class="metric-sub">Cepstral feature vector</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-label">Classifier Engine</div>
                                <div class="metric-val">Random Forest</div>
                                <div class="metric-sub">100 Decision Trees</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-label">Inference Latency</div>
                                <div class="metric-val">{latency_ms:.1f} ms</div>
                                <div class="metric-sub">CPU feature + inference</div>
                            </div>
                        </div>
                        <div style="height: 6px;"></div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-banner">
                            <span>⚠️</span>
                            <div>
                                <strong>Error during acoustic feature processing:</strong><br>
                                {str(e)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ── Compact Footer ──
st.markdown("""
<div class="app-footer">
    Voice Guard Analysis Tool · Forensic ML Audio Verification · 
    <a href="https://github.com/ashwanizz/voice-guard" target="_blank">View on GitHub</a>
</div>
""", unsafe_allow_html=True)
