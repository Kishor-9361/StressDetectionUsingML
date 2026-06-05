
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm
from model import MultimodalStressDetector

# --- Configuration ---
FACES_DATA_DIR = r"f:\Multimodal_stress_Detection\datasets\facesData"
MODEL_SAVE_PATH = "facial_expert_model.pkl"

def load_faces_data(data_dir):
    """
    Load facial images from the specified directory structure:
    data_dir/
      train/
        stress/
        nostress/
      test/
        stress/
        nostress/
    """
    print(f"Loading facial dataset from {data_dir}...")
    
    X = []
    y = []
    
    # Initialize detector to use its feature extraction logic
    detector = MultimodalStressDetector()
    
    # Categories
    categories = {'nostress': 0, 'stress': 1}
    
    # Iterate through Train and Test folders
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split} directory not found.")
            continue
            
        for label_name, label_val in categories.items():
            folder_path = os.path.join(split_dir, label_name)
            if not os.path.exists(folder_path):
                continue
                
            print(f"Processing {split}/{label_name}...")
            
            # Limit for demonstration/speed if needed, set to None for full
            files = os.listdir(folder_path)
            
            for file_name in tqdm(files):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, file_name)
                    
                    try:
                        features, _ = detector.extract_facial_features(file_path)
                        
                        if features is not None and not np.all(features == 0):
                            X.append(features)
                            y.append(label_val)
                    except Exception as e:
                        continue

    return np.array(X), np.array(y)

def train_facial_expert():
    # 1. Load Data (with caching)
    cache_path = "facial_features_cache.npz"
    if os.path.exists(cache_path):
        print(f"Loading features from cache {cache_path}...")
        data = np.load(cache_path)
        X, y = data['X'], data['y']
    else:
        X, y = load_faces_data(FACES_DATA_DIR)
        if len(X) > 0:
            print(f"Saving features to cache {cache_path}...")
            np.savez(cache_path, X=X, y=y)
    
    if len(X) == 0:
        print("Error: No data loaded. Check dataset path.")
        return

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # 2. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train Model (Reverting to Random Forest)
    print("\nTraining Facial Expert Model (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    print("\nEvaluating on Validation Set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Stress', 'Stress']))

    # 5. Save Model
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(clf, f)
        
    print("Done!")

if __name__ == "__main__":
    train_facial_expert()
