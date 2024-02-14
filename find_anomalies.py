from tensorflow.keras.models import load_model
import numpy as np
import pickle
import random
from extract import extract_images
import cv2
import os

timestamp = "14-02-2024-11-13-06"
dataset = os.path.join("output",f"{timestamp}.pickle")
model = os.path.join("model",f"{timestamp}.model")
anomaly_file = os.path.join("labels", f"{timestamp}.txt")

autoencoder = load_model(model)
images = pickle.loads(open(dataset, "rb").read())
anomaly_indexes = []
with open(anomaly_file, 'r') as f:
    for anomaly_index in f.readlines():
        anomaly_indexes.append(int(anomaly_index))
# def build_dataset_with_labels():
#     images, labels = extract_images()
#     valid_indexes = np.where(labels == 0) [0]
#     anomaly_indexes = np.where(labels == 1) [0]

#     random.shuffle(valid_indexes)
#     random.shuffle(anomaly_indexes)

#     anomalies = int(len(valid_indexes) * 0.01)
#     anomaly_indexes = anomaly_indexes[:anomalies]

#     valid_images = images[valid_indexes]
#     valid_labels = labels[valid_indexes]
#     anomaly_images = images[anomaly_indexes]
#     anomaly_labels = labels[anomaly_indexes]
#     images = np.vstack((valid_images, anomaly_images))
#     labels = np.concatenate((valid_labels, anomaly_labels))
#     randomise = np.arange(len(labels))
#     np.random.seed(77)
#     np.random.shuffle(randomise)
#     images = images[randomise]
#     labels = labels[randomise]
#     return images, labels, anomalies

# images, labels, anomaly_count = build_dataset_with_labels()
decoded = autoencoder.predict(images)
errors = []

for (image, reconstruction) in zip(images, decoded):
    if len(image.shape) == len(reconstruction.shape) + 1:
        image = image.reshape(image.shape[:-1])
    mse = np.mean((image - reconstruction)**2)
    errors.append(mse)

threshold = np.quantile(errors, 0.95)
anomalies = np.where(np.array(errors) >= threshold)[0]
anomaly_errors = [error for error in errors if error >= threshold]
print(f"mse threshold: {threshold}")
anomaly_count = len(anomaly_indexes)
print(f"{len(anomalies)} anomalies found of {len(images)} images and {anomaly_count} anomalies")
sum = 0
for anomaly in anomalies:
    if anomaly in anomaly_indexes:
        sum+=1
print(f"success rate {sum} out of {anomaly_count}")

outputs = None
for error, anomaly in sorted(zip(anomaly_errors, anomalies), reverse=True):
    original = (images[anomaly] * 255).astype("uint8")
    reconstruction = (decoded[anomaly] * 255).astype("uint8")
    if len(original.shape) == len(reconstruction.shape) +1:
        original = np.reshape(original, original.shape[:-1])
    output = np.hstack([original, reconstruction])
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack([outputs, output])
cv2.imwrite(os.path.join("anomalies",f"{timestamp}.png"), outputs)