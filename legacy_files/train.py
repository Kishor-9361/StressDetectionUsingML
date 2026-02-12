from voice import train_model
import numpy as np

model, label_encoder = train_model()
np.save('label_encoder_classes.npy', label_encoder.classes_)