"""
Training script for the Multimodal Stress Detection Model
This script trains the model using your dataset structure from the Jupyter notebook
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from model import MultimodalStressDetector
import os
import sys

# Set working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_dataset():
    """
    Load the dataset based on your existing structure
    Modify the paths according to your dataset location
    """
    print("Loading dataset...")
    
    # Load labels
    labels_path = 'Dataset/labels.csv'  # Update this path
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {os.path.abspath(labels_path)}")
        print("Please update the path in train_model.py")
        return None, None, None, None
    
    labels = pd.read_csv(labels_path, sep=",", header=0, index_col=0).dropna()
    print(f"Loaded {len(labels)} labels")
    
    # Load features for each modality
    phys_features_path = 'Feature Extraction/Features/all_physiological_features.csv'
    video_features_path = 'Feature Extraction/Features/video11tasks_aus_gaze_mean_std.csv'
    audio_features_path = 'Feature Extraction/Features/HCfeatures.csv'
    
    # Check if files exist
    if not os.path.exists(phys_features_path):
        print(f"Error: Physiological features not found at {os.path.abspath(phys_features_path)}")
        return None, None, None, None
    if not os.path.exists(video_features_path):
        print(f"Error: Video features not found at {os.path.abspath(video_features_path)}")
        return None, None, None, None
    if not os.path.exists(audio_features_path):
        print(f"Error: Audio features not found at {os.path.abspath(audio_features_path)}")
        return None, None, None, None
    
    print("Loading feature files...")
    x_phys = pd.read_csv(phys_features_path, sep=",", header=0, index_col=0)
    print(f"Loaded {len(x_phys)} physiological samples")
    
    x_video = pd.read_csv(video_features_path, sep=",", header=0, index_col=0)
    print(f"Loaded {len(x_video)} video samples")
    
    x_audio = pd.read_csv(audio_features_path, sep=",", header=None, index_col=0)
    print(f"Loaded {len(x_audio)} audio samples")
    
    # Process audio features if needed (from your notebook)
    # Fixed: Use assignment instead of inplace
    x_audio.index = [i.split('.')[0] for i in list(x_audio.index)]
    
    # Get common indices (tasks with all 3 modalities)
    common_idx = list(x_phys.index.intersection(x_video.index).intersection(x_audio.index).intersection(labels.index))
    
    print(f"\nFound {len(common_idx)} samples with all 3 modalities")
    
    if len(common_idx) == 0:
        print("Error: No common samples found across all modalities!")
        print(f"Phys samples: {len(x_phys)}")
        print(f"Video samples: {len(x_video)}")
        print(f"Audio samples: {len(x_audio)}")
        print(f"Label samples: {len(labels)}")
        return None, None, None, None
    
    # Filter to common indices
    x_phys = x_phys.loc[common_idx]
    x_video = x_video.loc[common_idx]
    x_audio = x_audio.loc[common_idx]
    y = labels.loc[common_idx]['binary-stress']
    
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Feature dimensions - Phys: {x_phys.shape[1]}, Video: {x_video.shape[1]}, Audio: {x_audio.shape[1]}")
    
    return x_phys.values, x_video.values, x_audio.values, y.values

def train_model_with_dataset():
    """
    Train the model using the actual dataset
    """
    # Load dataset
    X_phys, X_video, X_audio, y = load_dataset()
    
    if X_phys is None:
        return False
    
    # Split data
    print("\nSplitting data into train/test sets (80/20)...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_phys_train, X_phys_test = X_phys[train_idx], X_phys[test_idx]
    X_video_train, X_video_test = X_video[train_idx], X_video[test_idx]
    X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"\nFeature dimensions:")
    print(f"  Physiological: {X_phys_train.shape[1]}")
    print(f"  Video: {X_video_train.shape[1]}")
    print(f"  Audio: {X_audio_train.shape[1]}")
    
    # Apply SMOTE to balance training data
    # IMPORTANT: Need to stack all features together, apply SMOTE once, then split
    print("\nBalancing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    
    # Combine all features for SMOTE in the order: video, audio, phys
    X_combined_train = np.hstack([X_video_train, X_audio_train, X_phys_train])
    X_combined_train, y_balanced = smote.fit_resample(X_combined_train, y_train)
    
    # Split back into individual modalities after SMOTE
    n_video = X_video_train.shape[1]
    n_audio = X_audio_train.shape[1]
    n_phys = X_phys_train.shape[1]
    
    X_video_train_balanced = X_combined_train[:, :n_video]
    X_audio_train_balanced = X_combined_train[:, n_video:n_video+n_audio]
    X_phys_train_balanced = X_combined_train[:, n_video+n_audio:]
    
    print(f"Training set size after SMOTE: {len(y_balanced)}")
    print(f"Class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
    print(f"Balanced feature shapes:")
    print(f"  Video (facial): {X_video_train_balanced.shape}")
    print(f"  Audio (voice): {X_audio_train_balanced.shape}")
    print(f"  Physiological: {X_phys_train_balanced.shape}")
    
    # Initialize and train model
    print("\n" + "="*60)
    print("Initializing model...")
    model = MultimodalStressDetector()
    
    print("Training model (this may take a few minutes)...")
    print("="*60)
    # IMPORTANT: Order is X_facial (video), X_voice (audio), X_phys (physiological)
    model.train(X_video_train_balanced, X_audio_train_balanced, X_phys_train_balanced, y_balanced)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    print(f"\nTest set feature shapes:")
    print(f"  Physiological: {X_phys_test.shape}")
    print(f"  Video: {X_video_test.shape}")
    print(f"  Audio: {X_audio_test.shape}")
    
    correct = 0
    total = len(test_idx)
    predictions = []
    actuals = []
    
    print(f"\nPredicting {total} test samples...")
    for i in range(total):
        result = model.predict(
            facial_features=X_video_test[i],  # Video features for facial model
            voice_features=X_audio_test[i],    # Audio features for voice model
            phys_features=X_phys_test[i]      # Physiological features
        )
        
        predicted = 1 if result['predicted_class'] == 'Stress' else 0
        predictions.append(predicted)
        actuals.append(y_test[i])
        
        if predicted == y_test[i]:
            correct += 1
    
    accuracy = correct / total
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1-Score: {f1_score(actuals, predictions, average='weighted'):.2%}")
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, target_names=['No Stress', 'Stress']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(actuals, predictions))
    
    # Save model
    model_path = 'multimodal_stress_model.pkl'
    model.save_model(model_path)
    print(f"\n[SUCCESS] Model saved to {model_path}")
    
    return True

def create_demo_model():
    """
    Create a demo model with synthetic data for testing
    Use this if you don't have the full dataset yet
    """
    print("Creating demo model with synthetic data...")
    print("Note: This is for testing purposes only. Train with real data for production use.\n")
    
    # Generate synthetic data
    n_samples = 200
    
    # Facial features (Matches model.py: 84 features)
    X_facial = np.random.randn(n_samples, 84)
    
    # Voice features (Matches model.py: 140 features)
    X_voice = np.random.randn(n_samples, 140)
    
    # Physiological features (Matches model.py: 132 features)
    X_phys = np.random.randn(n_samples, 132)
    
    # Generate labels (binary: 0 = no stress, 1 = stress)
    y = np.random.randint(0, 2, n_samples)
    
    # Add some correlation to make it more realistic
    X_facial[y == 1, 0] += 0.5  # Higher values correlate with stress
    X_voice[y == 1, 5] += 0.5
    X_phys[y == 1, 2] += 0.5
    
    print(f"Generated {n_samples} synthetic samples")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Initialize and train model
    model = MultimodalStressDetector()
    
    print("\nTraining demo model...")
    model.train(X_facial, X_voice, X_phys, y)
    
    # Save model
    model_path = 'multimodal_stress_model.pkl'
    model.save_model(model_path)
    print(f"\n[SUCCESS] Demo model saved to {model_path}")
    print("\n[WARNING] Remember to train with real data before production use!")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Stress Detection Model Training")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Train with actual dataset (requires dataset files)")
    print("2. Create demo model with synthetic data (for testing)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        success = train_model_with_dataset()
        if not success:
            print("\n" + "="*60)
            print("[ERROR] Dataset loading failed. You can:")
            print("   - Check that all dataset files exist at the specified paths")
            print("   - Update the file paths in train_model.py")
            print("   - Or run option 2 to create a demo model for testing")
            print("="*60)
    elif choice == '2':
        create_demo_model()
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("You can now run app.py to start the Flask server")
    print("=" * 60)