import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from os import system
import matplotlib.pyplot as plt


BATCH_SIZE = 32

system('cls')

test_dir = './dataset/test_set'
# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2()

# Load test set
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_set = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(160, 160),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

class_names = list(test_set.class_indices.keys())
class_names.append('other')
print(class_names)

# Make predictions on test set
preds = model.predict(test_set)
print(preds)

# Get predicted labels
predicted_labels = np.argmax(preds, axis=1)
#151-268 = perro
#281-285 = gato
print(predicted_labels)

wrong_preds = list(filter(lambda x: x < 151 or (x > 268 and x < 281) or x > 285, predicted_labels))
import pandas as pd
classes_names_net = pd.read_json('./imagenet_class_index.json')
[print(classes_names_net[i][1]) for i in wrong_preds]
system('pause')
y_pred = [0 if pred >= 281 and pred <=285 else (1 if pred >= 151 and pred <=268 else 2) for pred in predicted_labels]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(confusion_matrix(y_true=test_set.labels, y_pred=y_pred))
print(classification_report(y_true=test_set.labels, y_pred=y_pred))
