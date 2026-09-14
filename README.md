# Voice Guard 🛡️
**Enterprise-Grade Audio Authentication & Deepfake Detection**

Voice Guard is a machine learning utility designed to detect synthetically generated audio (deepfakes). It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from `.wav` files and classifies them using a trained Random Forest model. 

Built as a 4th-year computer science micro-project, this tool provides a highly clean, native-feeling, academic-style dashboard built with Streamlit.

---

## Features
- **MFCC Feature Extraction:** Standardizes audio to exactly 16kHz and 5.0 seconds, then extracts 40 MFCCs as a biometric signature.
- **Random Forest Classifier:** A fast, interpretable model trained on diverse datasets (OpenAI, xTTS, VoiceBox, etc.) achieving high accuracy.
- **Academic Interface:** A distraction-free, professional, GitHub-style user interface.
- **Waveform Visualization:** Analyzes and visualizes the audio amplitude directly in the browser.

## Project Structure
```text
📦 Voice Guard
 ┣ 📂 data              # Raw and processed .wav files (ignored in git)
 ┣ 📂 models            # Pickled machine learning models (rf_model.pkl)
 ┣ 📜 app.py            # Main Streamlit web application
 ┣ 📜 extract_features.py # Script for generating MFCC arrays
 ┣ 📜 preprocess.py     # Script for audio standardization
 ┣ 📜 train.py          # Script for model training and evaluation
 ┗ 📜 requirements.txt  # Python dependencies
```

## Local Installation
1. Clone this repository to your local machine.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Dataset (Not Included)
Due to size constraints, the training audio data is not included in this repository. To retrain the model, place real `.wav` samples in `data/raw/real_samples/` and synthetic samples in folders like `data/raw/OpenAI/` or `data/raw/xTTS/`, then run the preprocessing and training scripts sequentially.
