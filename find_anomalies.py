from tensorflow.keras.models import load_model
import numpy as np
import pickle
import random
from extract import extract_images
import cv2
import os

timestamp = "21-02-2024-10-45-33"
dataset = os.path.join("output",f"{timestamp}.pickle")
model = os.path.join("model",f"{timestamp}.model")
anomaly_file = os.path.join("labels", f"{timestamp}.txt")

autoencoder = load_model(model)
images = pickle.loads(open(dataset, "rb").read())
anomaly_indexes = []
with open(anomaly_file, 'r') as f:
    for anomaly_index in f.readlines():
        anomaly_indexes.append(int(anomaly_index))

decoded = autoencoder.predict(images)
errors = []
pxl_errors = []

for (image, reconstruction) in zip(images, decoded):
    if len(image.shape) == len(reconstruction.shape) + 1:
        image = image.reshape(image.shape[:-1])
    mse = np.mean((image - reconstruction)**2)
    pxl_error = np.mean((image - reconstruction)**2, axis=2)
    errors.append(mse)
    pxl_errors.append(pxl_error)

threshold = np.quantile(errors, 0.98)
anomalies = np.where(np.array(errors) >= threshold)[0]
anomaly_errors = [error for error in errors if error >= threshold]
anomaly_pxl_errors = [pxl_error for pxl_error, error in zip(pxl_errors,errors) if error>=threshold]
print(f"mse threshold: {threshold}")
anomaly_count = len(anomaly_indexes)
print(f"{len(anomalies)} anomalies found of {len(images)} images and {anomaly_count} anomalies")
sum = 0
for anomaly in anomalies:
    if anomaly in anomaly_indexes:
        sum+=1
print(f"success rate {sum} out of {anomaly_count}")

outputs = None
for error, anomaly, pxl_error in sorted(zip(anomaly_errors, anomalies, anomaly_pxl_errors), reverse=True):
    original = (images[anomaly] * 255).astype("uint8")
    reconstruction = (decoded[anomaly] * 255).astype("uint8")
    heatmap = np.expand_dims(np.round((pxl_error/np.max(pxl_error))*255), axis=-1).astype('uint8')
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = np.hstack([original, reconstruction,
                         heatmap_img
                         ])
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack([outputs, output])
    print(anomaly in anomaly_indexes)
cv2.imwrite(os.path.join("anomalies",f"{timestamp}.png"), outputs)