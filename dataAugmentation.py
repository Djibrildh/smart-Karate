'''
il faut : 
- Multiplier par 4 les blocage_droit
- Multiplier par 2 les blocage_gauche
- revisualiser les data
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import random
import shutil

def add_gaussian_noise(image, mean=0.5, sigma=2.2):
    """ Add Gaussian noise to an image """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255)

def random_rotation(image, angle):
    """ Randomly rotate the image within the given angle range """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def zoom_image(image, zoom_factor = 1.5):

    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    center_y, center_x = new_height // 2, new_width // 2
    crop_y1, crop_y2 = center_y - height // 2, center_y + height // 2
    crop_x1, crop_x2 = center_x - width // 2, center_x + width // 2
    
    zoomed_image = resized_image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return zoomed_image


def augmentDataA(img, angle=10):
    a = random_rotation(img, angle)
    b = random_rotation(img, -angle)
    c = add_gaussian_noise(img)
    d = zoom_image(img)
    return [img,a,b,c,d]

def augmentDataB(img, angle=10):
    a = random_rotation(img, angle)
    b = zoom_image(img)
    return [img,a,b]

def findData(datapath, imagepath, output_folder):
    print("here")
    with open(datapath, "r") as f:
        data = json.load(f)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    liste_image = os.listdir(imagepath)
    
    for image_name in liste_image:
        image_path = os.path.join(imagepath, image_name)
        
        normalized_path = os.path.normpath(image_path)
        print("here2")
        if normalized_path in data:
            if data[normalized_path]["label"] == "blocage_gauche":
                img = cv2.imread(image_path)               
                im, a, b= augmentDataB(img)          
                output_path = os.path.join(output_folder, f"augmentedA_{image_name}")               
                success = cv2.imwrite(output_path, a)               
                if success:
                    print(f"Image enregistrée dans {output_folder}")
                output_path = os.path.join(output_folder, f"augmentedB_{image_name}")
                success = cv2.imwrite(output_path, b)
                if success:
                    print(f"Image enregistrée dans {output_folder}")
                
            elif data[normalized_path]["label"] == "blocage_droit":
                img = cv2.imread(image_path)
                im, a, b, c, d= augmentDataA(img)
                output_path = os.path.join(output_folder, f"augmented_{image_name}")
                success = cv2.imwrite(output_path, a)
                
                if success:
                    print(f"Image enregistrée dans {output_folder}")
                # Chemin de sortie pour l'image augmentée
                output_path = os.path.join(output_folder, f"augmentedB_{image_name}")
                # Enregistrer l'image augmentée
                success = cv2.imwrite(output_path, b)
                if success:
                    print(f"Image enregistrée dans {output_folder}")
                # Chemin de sortie pour l'image augmentée
                output_path = os.path.join(output_folder, f"augmentedC_{image_name}")
                # Enregistrer l'image augmentée
                success = cv2.imwrite(output_path, c)
                if success:
                    print(f"Image enregistrée dans {output_folder}")
                # Chemin de sortie pour l'image augmentée
                output_path = os.path.join(output_folder, f"augmentedD_{image_name}")
                # Enregistrer l'image augmentée
                success = cv2.imwrite(output_path, d)
                if success:
                    print(f"Image enregistrée dans {output_folder}")
        else:
            print(f"L'image {normalized_path} n'a pas été trouvée dans le JSON.")

def undersample_class(imagepath, output_folder, label, target_count):
    liste_image = os.listdir(imagepath)
    images_to_keep = random.sample(liste_image, target_count)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in images_to_keep:
        image_path = os.path.join(imagepath, image_name)
        output_path = os.path.join(output_folder, image_name)
        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)


def mergeData(datapath,data_augmented_path,output_folder):

    with open(datapath, "r") as f:
        data = json.load(f)
    
    with open(data_augmented_path, "r") as f:
        data_bis = json.load(f)
    
    merged_dict = {**data, **data_bis}


    with open(output_folder, "w") as json_file:
        json.dump(merged_dict, json_file, indent=4)
    
    print("Dataset enregistré dans : ", output_folder)


def findData(datapath, imagepath, output_folder, label):
    with open(datapath, "r") as f:
        data = json.load(f)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    liste_image = os.listdir(imagepath)
    
    for image_name in liste_image:
        image_path = os.path.join(imagepath, image_name)
        normalized_path = os.path.normpath(image_path)
        
        if normalized_path in data:
            if data[normalized_path]["label"] == label:
                img = cv2.imread(image_path)
                
                if label == "blocage_gauche":
                    im, a, b = augmentDataB(img)
                    
                    output_path = os.path.join(output_folder, f"augmentedA_{image_name}")
                    cv2.imwrite(output_path, a)
                    
                    output_path = os.path.join(output_folder, f"augmentedB_{image_name}")
                    cv2.imwrite(output_path, b)
                
                elif label == "blocage_droit":
                    im, a, b, c, d = augmentDataA(img)
                    
                    output_path = os.path.join(output_folder, f"augmentedA_{image_name}")
                    cv2.imwrite(output_path, a)
                    
                    output_path = os.path.join(output_folder, f"augmentedB_{image_name}")
                    cv2.imwrite(output_path, b)
                    
                    output_path = os.path.join(output_folder, f"augmentedC_{image_name}")
                    cv2.imwrite(output_path, c)
                    
                    output_path = os.path.join(output_folder, f"augmentedD_{image_name}")
                    cv2.imwrite(output_path, d)
        else:
            print(f"L'image {normalized_path} n'a pas été trouvée dans le JSON.")

'''# Suréchantillonner blocage_droit
findData("dataReduce", "ImagesReduce", "ImagesReduce", "blocage_droit")

# Suréchantillonner blocage_gauche
findData("dataReduce", "ImagesReduce", "ImagesReduce", "blocage_gauche")'''

def undersample_class(imagepath, output_folder, target_count):
    liste_image = os.listdir(imagepath)
    images_to_keep = random.sample(liste_image, target_count)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in images_to_keep:
        image_path = os.path.join(imagepath, image_name)
        output_path = os.path.join(output_folder, image_name)
        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)

import json
import random

def limit_samples_per_label(json_path, output_json_path, max_samples_per_label=250):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    label_to_images = {}
    
    for image_path, image_data in data.items():
        label = image_data["label"]
        
        if label not in label_to_images:
            label_to_images[label] = []
        
        label_to_images[label].append((image_path, image_data))
    
    output_data = {}
    
    for label, images in label_to_images.items():
        if len(images) > max_samples_per_label:
            selected_images = random.sample(images, max_samples_per_label)
        else:
            selected_images = images  
        
        for image_path, image_data in selected_images:
            output_data[image_path] = image_data
    
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Le fichier JSON avec un maximum de {max_samples_per_label} échantillons par label a été enregistré dans {output_json_path}.")

#limit_samples_per_label("dataSetAugmentedBis", "dataSetAugmented", max_samples_per_label=410)