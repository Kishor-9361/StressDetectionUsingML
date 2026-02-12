import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision (if GPU available)
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled")
except:
    print("⚠️ Mixed precision not supported, continuing with default precision")

class StressDetectionModel:
    def __init__(self, img_size=(128, 128), batch_size=64):  # Increased batch size
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['No Stress', 'Stress']

    def create_data_generators(self, train_dir, test_dir):
        """Create training, validation, and test datasets using tf.data"""
        print("🔄 Creating data generators...")

        # Minimal augmentation for speed
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,  # Reduced from 10
            width_shift_range=0.05,  # Reduced from 0.1
            height_shift_range=0.05,  # Reduced from 0.1
            horizontal_flip=True,
            validation_split=0.2
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True
        )

        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        # Convert to tf.data for faster loading
        def gen_to_tfdata(generator):
            dataset = tf.data.Dataset.from_generator(
                lambda: generator,
                output_types=(tf.float32, tf.float32),
                output_shapes=([None, *self.img_size, 3], [None])
            )
            dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
            return dataset

        train_dataset = gen_to_tfdata(train_generator)
        val_dataset = gen_to_tfdata(val_generator)
        test_dataset = gen_to_tfdata(test_generator)

        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")

        return train_dataset, val_dataset, test_dataset, train_generator, test_generator

    def build_model(self):
        """Use MobileNetV2 for faster training and better accuracy"""
        print("🧠 Building MobileNetV2-based model...")

        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(input_shape=(*self.img_size, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze base model for speed

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid', dtype='float32')  # Mixed precision compatibility
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("✅ Model built successfully!")
        return model

    def train_model(self, train_dataset, val_dataset, train_generator, epochs=10):  # Reduced epochs
        """Train model with early stopping"""
        print("🚀 Starting model training...")

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),  # Tighter patience
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
        ]

        self.history = self.model.fit(
            train_dataset,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=train_generator.samples // self.batch_size // 5,
            callbacks=cb,
            verbose=1
        )

        print("✅ Training completed!")
        return self.history

    def evaluate_model(self, test_dataset, test_generator):
        """Evaluate trained model"""
        print("📊 Evaluating model on test data...")
        predictions = self.model.predict(test_dataset)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_generator.classes

        accuracy = accuracy_score(true_classes, predicted_classes)
        print(f"\n✅ Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        return accuracy

    def save_model(self, filepath='final_stress_detection_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"💾 Model saved successfully as {filepath}")
        else:
            print("⚠️ No model found to save.")

    def predict_single_image(self, image_path):
        """Predict stress or no stress for one image"""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            img = Image.open(image_path).convert('RGB').resize(self.img_size)
            img = np.array(img)
            img = preprocess_input(img)  # MobileNetV2 preprocessing
            img = np.expand_dims(img, axis=0)
            prediction = self.model.predict(img)[0][0]

            return {
                "predicted_class": "Stress" if prediction > 0.5 else "No Stress",
                "confidence": float(prediction if prediction > 0.5 else 1 - prediction)
            }
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

# -----------------------------
# 🚀 MAIN TRAINING FUNCTION
# -----------------------------
def train_model():
    print("🎯 Starting Stress Detection Model...")

    # Initialize model
    stress_model = StressDetectionModel(img_size=(128, 128), batch_size=64)

    # Update these paths as per your folder structure
    train_dir = r"D:/stress/stress-detection/src/facesData/train"
    test_dir = r"D:/stress/stress-detection/src/facesData/test"

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("❌ Dataset directories not found!")
        print(f"Expected paths:\n{train_dir}\n{test_dir}")
        return None

    # Create data generators
    train_dataset, val_dataset, test_dataset, train_gen, test_gen = stress_model.create_data_generators(train_dir, test_dir)

    # Build, train and evaluate
    stress_model.build_model()
    stress_model.train_model(train_dataset, val_dataset, train_gen, epochs=10)
    stress_model.evaluate_model(test_dataset, test_gen)
    stress_model.save_model("final_stress_detection_model.h5")

    print("\n🏁 All steps completed successfully!")
    return stress_model

# -----------------------------
# 🏃 MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    model = train_model()