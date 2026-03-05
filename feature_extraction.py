import cv2
import numpy as np

def extract_features(image):

    if image is None:
        return None

    # Resize
    image = cv2.resize(image, (224,224))

    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # RGB mean
    r_mean = np.mean(rgb[:,:,0])
    g_mean = np.mean(rgb[:,:,1])
    b_mean = np.mean(rgb[:,:,2])

    # HSV mean
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])

    features = [
        r_mean, g_mean, b_mean,
        h_mean, s_mean, v_mean
    ]

    return np.array(features)