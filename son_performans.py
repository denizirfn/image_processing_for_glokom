import cv2
import numpy as np
import os
import pandas as pd


def correct_illumination(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = min(gray.shape) // 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    foreground = cv2.subtract(gray, background)
    corrected = cv2.add(foreground, 50)
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)


def remove_blood_vessels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)
    kernel = np.ones((3, 3), np.uint8)
    vessel_mask = cv2.morphologyEx(enhanced_gray, cv2.MORPH_GRADIENT, kernel)
    _, vessel_mask = cv2.threshold(vessel_mask, 19, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, vessel_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA), vessel_mask


def multi_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, cup_mask = cv2.threshold(gray_normalized, 150, 255, cv2.THRESH_BINARY)
    _, disc_mask = cv2.threshold(gray_normalized, 100, 255, cv2.THRESH_BINARY)
    return disc_mask, cup_mask


def calculate_disc_cup_ratio(disc_mask, cup_mask):
    disc_area = np.sum(disc_mask == 255)
    cup_area = np.sum(cup_mask == 255)
    return cup_area / disc_area if disc_area != 0 else 0


def glaucoma_detection(disc_mask, cup_mask):
    ratio = calculate_disc_cup_ratio(disc_mask, cup_mask)
    return "Glokom mevcut" if ratio > 0.5 else "Glokom değil"


def process_images(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            corrected_image = correct_illumination(image)
            vessel_removed_image, _ = remove_blood_vessels(corrected_image)
            disc_mask, cup_mask = multi_thresholding(vessel_removed_image)
            prediction = glaucoma_detection(disc_mask, cup_mask)
            results.append((filename, prediction))

    return results


def evaluate_model(results):
    df = pd.DataFrame(results, columns=['filename', 'prediction'])
    TP = np.sum(df['prediction'] == "Glokom mevcut")
    FN = np.sum(df['prediction'] == "Glokom değil")
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    print(f"Model Recall (Hassasiyet): {recall:.2f}")
    return df


# Örnek kullanım
folder_path = 'glaucoma'
results = process_images(folder_path)
evaluate_model(results)
