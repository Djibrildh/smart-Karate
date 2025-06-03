import os
import cv2

def datasetMaker(datapath, output_folder):
    
    # Vérifie si le dossier de sortie existe, sinon le crée
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    list_dirs = os.listdir(datapath)
    saved_count = 0

    for subdir in list_dirs:
        subdir_path = os.path.join(datapath, subdir)  # Chemin du sous-dossier

        if os.path.isdir(subdir_path):  # Vérifie si c'est bien un dossier
            images = sorted(os.listdir(subdir_path))

            for img_name in images:
                img_path = os.path.join(subdir_path, img_name)  # Chemin complet de l'image
                image = cv2.imread(img_path)

                if image is not None:  # Vérifie si l'image est chargée correctement
                    output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                    success = cv2.imwrite(output_path, image)  # 📸 Enregistre l'image

                    if success:
                        print(f"Image enregistrée : {output_path}")
                    else:
                        print(f"Échec de l'enregistrement : {output_path}")

                    saved_count += 1
                else:
                    print(f"Erreur : Impossible de lire {img_path}")


output_folder = "Images"
datapath = "data_set"