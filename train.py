import matplotlib
matplotlib.use("Agg")

from autoencoder import Autoencoder
from extract import extract_images
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
from datetime import datetime
import os

now = datetime.now()
date_time = now.strftime("%d-%m-%Y-%H-%M-%S")

def build_dataset(data: np.ndarray, labels: np.ndarray, valid_label: int =1, anomaly_label: int =0, contamination: float = 0.01, seed: int = 77) -> np.ndarray:
    valid_indexes = np.where(labels == valid_label) [0]
    anomaly_indexes = np.where(labels == anomaly_label) [0]

    random.seed(seed)
    random.shuffle(valid_indexes)
    random.shuffle(anomaly_indexes)

    anomalies = int(len(valid_indexes) * contamination)
    anomaly_indexes = anomaly_indexes[:anomalies]

    valid_images = data[valid_indexes]
    anomaly_images = data[anomaly_indexes]

    labels_list = []
    for image in valid_images:
        labels_list.append(valid_label)
    for image in anomaly_images:
        labels_list.append(anomaly_label)
    labels_array = np.asarray(labels_list)
    images = np.vstack((valid_images, anomaly_images))
    np.random.seed(seed)
    randomise = np.arange(len(images))
    np.random.shuffle(randomise)
    images = images[randomise]
    labels_array = labels_array[randomise]
    with open(os.path.join('labels',f"{date_time}.txt"),'w+') as f:
        for label in range(len(labels_array)):
            if labels_array[label] == anomaly_label:
                f.write(f"{label}\n")
    return images

def show_predictions(decoded: np.ndarray, gt: np.ndarray, samples: int=10) -> np.ndarray:
    for sample in range(0, samples):
        original = (gt[sample] * 255).astype("uint8")
        reconstructed = (decoded[sample] * 255).astype("uint8")
        output = np.hstack([original, reconstructed])
        if sample == 0:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])
    return outputs

Epochs = 125
Init_LR = 1e-3
batch_size = 32

print("loading dataset")
data, labels = extract_images()
print("creating dataset")
images = build_dataset(data, labels, valid_label=1, anomaly_label=0, contamination=0.01)

images = images.astype("float32")/255.0
train_x, test_x = train_test_split(images, test_size=0.2, random_state=77)

print("building autoencoder")
autoencoder = Autoencoder.build(400,400,3, (16, 8), 3, 2, 16)
optimiser = Adam(learning_rate = Init_LR, decay = Init_LR/Epochs)
autoencoder.compile(loss = 'mae', optimizer = optimiser)

model = autoencoder.fit(train_x, train_x,
                        validation_data = (test_x, test_x),
                        epochs = Epochs,
                        batch_size = batch_size)
print("running predictions")
decoded = autoencoder.predict(test_x)
vis = show_predictions(decoded, test_x)
cv2.imwrite(os.path.join("reconstruction", f"{date_time}.png"), vis)

N = np.arange(0, Epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, model.history["loss"], label = "train_loss")
plt.plot(N, model.history["val_loss"], label = "val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join("loss",f"{date_time}.jpg"))

print("saving data")
dataset = os.path.join("output",f"{date_time}.pickle")
f = open(dataset, 'wb')
f.write(pickle.dumps(images))
f.close()

print("saving autoencoder")
model_out = os.path.join("model", f"{date_time}.model")
autoencoder.save(model_out, save_format='h5')