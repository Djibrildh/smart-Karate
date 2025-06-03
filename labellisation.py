import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
from genericpath import exists
import json

def extractLandmarksfromImage(img, pose):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append((lm.x,lm.y,lm.z,lm.visibility))
        return landmarks
    else:
        return None

def calculate_angle(a, b, c):
    """
    Calcule l'angle entre trois points a, b, c.
    a, b, c sont des tuples (x, y).
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)  # En radians
    return np.degrees(angle)  # Convertit en degrés

def classify_pose(landmarks):
    LEFT_WRIST = 15
    LEFT_HIP = 23
    LEFT_SHOULDER = 11
    RIGHT_WRIST = 16
    RIGHT_HIP = 24
    RIGHT_SHOULDER = 12

    # Extraction des positions
    lw, lh, ls = landmarks[LEFT_WRIST], landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER]
    rw, rh, rs = landmarks[RIGHT_WRIST], landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER]

    # Calcul des angles avec la hanche
    left_arm_angle = calculate_angle(lw[:2], lh[:2], ls[:2])  # Angle bras gauche
    right_arm_angle = calculate_angle(rw[:2], rh[:2], rs[:2])  # Angle bras droit

    #print(f"Angle bras gauche: {left_arm_angle}, bras droit: {right_arm_angle}")

    # 🔍 Détection de blocage
    if 45 <= left_arm_angle <= 100:  
        return "blocage_gauche"
    if 45 <= right_arm_angle <= 100:  
        return "blocage_droit"

    # 🔍 Ajustement des seuils
    if 10 <= left_arm_angle or right_arm_angle <= 40:
        if left_arm_angle > right_arm_angle : 
            return "coup_de_poing_gauche"
        else :
            return "coup_de_poing_droit"
        
    return None


def labelisationTest(image,output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    data = {}

    img = cv2.imread(image)
    
    image_landmarks = extractLandmarksfromImage(img,pose)

    if image_landmarks:
        label = classify_pose(image_landmarks)
        if label is not None:
            data[image] = {"landmarks" : image_landmarks, "label" : label}
    
    with open(output_folder, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Dataset enregistré dans : ", output_folder)


def labelisation(datapath,output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    data = {}

    images = sorted(os.listdir(datapath))

    for image in images:
        image = os.path.join(datapath,image)
        img = cv2.imread(image)
        
        image_landmarks = extractLandmarksfromImage(img,pose)

        if image_landmarks:
            label = classify_pose(image_landmarks)
            if label is not None:
                data[image] = {"landmarks" : image_landmarks, "label" : label}
    
    with open(output_folder, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Dataset enregistré dans : ", output_folder)

def augmentedLabelisation(datapath,output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    data = {}

    images = sorted(os.listdir(datapath))

    for image in images:
        image = os.path.join(datapath,image)
        img = cv2.imread(image)
        
        image_landmarks = extractLandmarksfromImage(img,pose)

        if image_landmarks:
            label = classify_pose(image_landmarks)
            if label is not None:
                if label == "blocage_droit" or label == "blocage_gauche":
                    data[image] = {"landmarks" : image_landmarks, "label" : label}

    with open(output_folder, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Dataset enregistré dans : ", output_folder)