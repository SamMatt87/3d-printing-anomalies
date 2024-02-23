from PIL import Image
import cv2
import numpy as np
import os
from typing import Tuple, List

def extract_images() -> Tuple[np.ndarray, np.ndarray]:
    data_directory = os.path.join(os.getcwd(), "archive")
    subfolders = [folder for folder in os.listdir(data_directory)]
    images: List[np.ndarray] = []
    labels: List[int] = []
    for subfolder in subfolders:
        for image in os.listdir(os.path.join(data_directory,subfolder)):
            img = cv2.imread(os.path.join(data_directory, subfolder, image))
            img = cv2.resize(img, (400,400))
            images.append(np.asarray(img))
            if subfolder.startswith('no_'):
                labels.append(1)
            else:
                labels.append(0)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels

if __name__ == "__main__":
    images, labels = extract_images()
    print(images.shape)
    print(labels.shape)