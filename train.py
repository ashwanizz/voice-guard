import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Configuration
FEATURES_DIR = os.path.join('data', 'features')
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.pkl')

def train_model():
    print("Loading features...")
    X_path = os.path.join(FEATURES_DIR, 'X.npy')
    y_path = os.path.join(FEATURES_DIR, 'y.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Features not found in {FEATURES_DIR}. Run extract_features.py first.")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    print("\n--- Classification Report ---")
    target_names = ['Real (0)', 'Fake (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
