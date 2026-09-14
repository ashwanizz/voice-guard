import os
import glob
import librosa
import numpy as np

# Configuration
PROCESSED_DIR = os.path.join('data', 'processed')
FEATURES_DIR = os.path.join('data', 'features')
SR = 16000
N_MFCC = 40

def extract_features():
    X = []
    y = []
    
    # Process Real Audio
    real_dir = os.path.join(PROCESSED_DIR, 'real')
    real_files = glob.glob(os.path.join(real_dir, '**', '*.wav'), recursive=True)
    
    print(f"Found {len(real_files)} real audio files.")
    for file_path in real_files:
        try:
            # We already resampled in preprocess, but we can load with sr=None
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            # Take the mean across the time axis
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            X.append(mfcc_mean)
            y.append(0)  # Label 0 for real
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Process Fake Audio
    fake_dir = os.path.join(PROCESSED_DIR, 'fake')
    fake_files = glob.glob(os.path.join(fake_dir, '**', '*.wav'), recursive=True)
    
    print(f"Found {len(fake_files)} fake audio files.")
    for file_path in fake_files:
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            X.append(mfcc_mean)
            y.append(1)  # Label 1 for fake
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Save features
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, 'X.npy'), X)
    np.save(os.path.join(FEATURES_DIR, 'y.npy'), y)
    
    print(f"Feature extraction complete.")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Saved to {FEATURES_DIR}")

if __name__ == "__main__":
    extract_features()
