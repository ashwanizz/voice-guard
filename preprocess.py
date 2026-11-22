import os
import glob
import librosa
import soundfile as sf
import numpy as np

# Configuration
RAW_DIR = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')
TARGET_SR = 16000
TARGET_DURATION = 5.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

REAL_FOLDER = 'real_samples'
FAKE_FOLDERS = [
    'FlashSpeech', 'NaturalSpeech3', 'OpenAI', 'PromptTTS2', 
    'VALLE', 'VoiceBox', 'seedtts_files', 'xTTS', 'fake_indian_tts'
]

def preprocess_audio(file_path, output_path):
    try:
        # Load audio, standardizing sample rate
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # Pad or truncate
        if len(y) > TARGET_SAMPLES:
            # Truncate
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            # Pad with silence (zeros)
            padding = TARGET_SAMPLES - len(y)
            y = np.pad(y, (0, padding), 'constant')
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed audio
        sf.write(output_path, y, TARGET_SR)
        print(f"Processed: {file_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not os.path.exists(RAW_DIR):
        print(f"Error: {RAW_DIR} does not exist. Please create it and add .wav files.")
        return

    # Find all .wav files in raw directory (including subdirectories)
    wav_files = glob.glob(os.path.join(RAW_DIR, '**', '*.wav'), recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {RAW_DIR}.")
        return

    print(f"Found {len(wav_files)} .wav files. Starting preprocessing...")
    
    for file_path in wav_files:
        # Determine if it's real or fake based on the folder name
        parts = os.path.normpath(file_path).split(os.sep)
        
        # Check if any part of the path matches our known folders
        label = None
        if REAL_FOLDER in parts:
            label = 'real'
        else:
            for fake_folder in FAKE_FOLDERS:
                if fake_folder in parts:
                    label = 'fake'
                    break
                    
        if label is None:
            print(f"Skipping {file_path}: Doesn't belong to known real/fake folders.")
            continue
            
        filename = os.path.basename(file_path)
        output_path = os.path.join(PROCESSED_DIR, label, filename)
        
        # Handle duplicate filenames by appending a unique identifier if necessary
        # For simplicity, we assume filenames are unique enough or we overwrite
        # Let's add the folder name to the filename to avoid collisions from different fake folders
        if label == 'fake':
            # find which fake folder it came from
            folder_name = next(f for f in FAKE_FOLDERS if f in parts)
            name, ext = os.path.splitext(filename)
            filename = f"{folder_name}_{name}{ext}"
            output_path = os.path.join(PROCESSED_DIR, label, filename)
            
        preprocess_audio(file_path, output_path)

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
