import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import sounddevice as sd
from scipy.io import wavfile

# --- Configuration ---
base_dataset_path = r"D:/stress/stress-detection/src/audio"

emotion_map = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}
intensity_map = {1: 'normal', 2: 'strong'}

# --- Data Collection ---
def collect_data():
    all_audio_data = []
    for actor_id in range(1, 25):
        actor_folder_name = f'Actor_{actor_id:02d}'
        actor_folder_path = os.path.join(base_dataset_path, actor_folder_name)
        if not os.path.exists(actor_folder_path):
            print(f"Warning: Actor folder not found: {actor_folder_path}")
            continue
        for filename in os.listdir(actor_folder_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(actor_folder_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        pass
                except (OSError, IOError) as e:
                    print(f"Skipping invalid file: {filepath} ({e})")
                    continue
                parts = filename.split('-')
                if len(parts) != 7:
                    print(f"Skipping malformed filename: {filename}")
                    continue
                try:
                    modality = int(parts[0])
                    vocal_channel = int(parts[1])
                    emotion_id = int(parts[2])
                    intensity_id = int(parts[3])
                    statement = int(parts[4])
                    repetition = int(parts[5])
                    actor_num = int(parts[6].split('.')[0])
                except ValueError:
                    print(f"Skipping filename with invalid format: {filename}")
                    continue
                emotion_label = emotion_map.get(emotion_id, 'unknown')
                intensity_label = intensity_map.get(intensity_id, 'unknown')
                gender = 'male' if actor_num % 2 != 0 else 'female'
                if emotion_id in [5, 6]:
                    stress_label = 'high_stress' if intensity_id == 2 else 'moderate_stress'
                elif emotion_id in [4, 7]:
                    stress_label = 'moderate_stress'
                elif emotion_id == 8:
                    stress_label = 'low_stress'
                elif emotion_id in [1, 2, 3]:
                    stress_label = 'non_stress'
                else:
                    stress_label = 'unknown'
                all_audio_data.append({
                    'filename': filename,
                    'filepath': filepath,
                    'modality': modality,
                    'vocal_channel': vocal_channel,
                    'emotion_id': emotion_id,
                    'emotion_label': emotion_label,
                    'intensity_id': intensity_id,
                    'intensity_label': intensity_label,
                    'statement': statement,
                    'repetition': repetition,
                    'actor_id': actor_num,
                    'gender': gender,
                    'stress_label': stress_label
                })
    df_metadata = pd.DataFrame(all_audio_data)
    print("--- Metadata DataFrame Created ---")
    print(f"Total audio files found: {len(df_metadata)}")
    print(df_metadata.head())
    print("\nValue counts for stress labels:")
    print(df_metadata['stress_label'].value_counts())
    df_metadata.to_csv('ravdess_stress_metadata.csv', index=False)
    print("\nMetadata saved to ravdess_stress_metadata.csv")
    return df_metadata

# --- Feature Extraction ---
def extract_mfcc(file_path, n_mfcc=13, max_length=200):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        return mfcc
    except Exception as e:
        print(f"Error extracting MFCC from {file_path}: {e}")
        return None

# --- Train Model ---
def train_model():
    df_metadata = collect_data()
    features = []
    labels = []
    print("\n--- Extracting MFCC Features ---")
    for index, row in df_metadata.iterrows():
        if row['stress_label'] != 'unknown':
            mfcc = extract_mfcc(row['filepath'])
            if mfcc is not None:
                features.append(mfcc)
                labels.append(row['stress_label'])
    if not features:
        raise ValueError("No valid features extracted. Check your dataset files.")
    features = np.array(features)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    encoded_labels = to_categorical(encoded_labels, num_classes=num_classes)
    features = features[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    print(f"Features shape: {features.shape}")
    print(f"Classes: {label_encoder.classes_}")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(features.shape[1], features.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("\n--- Training Model ---")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save('stress_detection_model.h5')
    print("Model saved to stress_detection_model.h5")
    return model, label_encoder

# --- Inference Functions ---
def predict_stress_from_file(file_path, model, label_encoder, n_mfcc=13, max_length=200):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        mfcc = extract_mfcc(file_path, n_mfcc, max_length)
        if mfcc is None:
            raise ValueError("Failed to extract MFCC features")
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def record_and_predict(duration=3, fs=22050, model=None, label_encoder=None):
    try:
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        temp_file = 'temp_recording.wav'
        wavfile.write(temp_file, fs, recording.flatten())
        print("Recording saved temporarily.")
        stress_level = predict_stress_from_file(temp_file, model, label_encoder)
        os.remove(temp_file)
        return stress_level
    except Exception as e:
        print(f"Error during recording: {e}")
        return None