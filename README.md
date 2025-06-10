# Kata Intelligent

## **Présentation du projet**

Dans le cadre de mon parcours en Master Ingénierie et Intelligence Artificielle à l’Université Paris 8, j’ai réalisé ce projet afin d’explorer l’apprentissage automatique appliqué à la reconnaissance de mouvements. L’objectif est de concevoir un **modèle capable de reconnaître automatiquement des mouvements spécifiques issus d’un kata de karaté** à l’aide d’un réseau de neurones de type MLP (Multi-Layer Perceptron).

Les données utilisées sont constituées d’**images extraites de vidéos de katas**, annotées selon différents types de mouvements (blocages, coups de poing, etc.).
Pour des raisons de droits d'auteur, je ne mets pas en ligne la base utiliser, car toutes les vidéos ne sont pas libres d'exploitation. 

## **Objectifs**
1. Prétraiter les images pour les adapter à l’apprentissage (mise à l’échelle, encodage, équilibrage).
2. Concevoir un modèle MLP pour classifier les types de mouvements :
   - **Blocage gauche / droit**
   - **Coup de poing gauche / droit**
3. Évaluer les performances du modèle à l’aide de métriques standards (précision, perte, matrice de confusion).


## **Structure du projet**

- `model/` : stocke tous les modèles entraînés (`.keras`).
- `main.ipynb` : notebook contenant une première exploration de mediapipe.
- `datasetMaker.py` : code permettant de construire le dataset avec une standardisation du nom des images ou vidéos.
- `labellisation.py` : code permettant de labelliser les images. 
- `training.py` : code contenant une première exploration de l'entrainement d'un modèle MLP.
- `dataAugmentation.py` : fichier permettant de modifier la base de données en augmentant les images. 
- `augmentationVisualisation.ipynb` : notebook permettant de visualiser l'augmentation effectué.
- `trainingDataMLP.ipynb` : notebook contenant l'entraînement sur un jeu de donnée.
- `trainingAugmentedMLP.ipynb` : notebook contenant l'entraînement sur un jeu de donnée.
- `trainingReduceMLP.ipynb` : notebook contenant l'entraînement sur un jeu de donnée.
- `trainingfinal.ipynb` :  notebook contenant l'entraînement sur un jeu de donnée.
- `README.md` : Présentation du projet.
- `requirements.txt` : Liste des bibliothèques utilisées.

## **Prérequis**

- Python 3.9 ou version supérieure jusque python 3.12.8.
- Un environnement virtuel est recommandé (`venv` ou `conda`).
- Librairies nécessaires listées dans `requirements.txt`.
- IDE compatible (Jupyter Notebook, Google Colab, etc.).

## **Étapes du projet**

1. **Prétraitement :**
   - Redimensionnement des images.
   - Encodage des étiquettes.
   - Split en jeux d'entraînement et de validation.
   - Éventuelles augmentations (zoom, flips, etc.).

2. **Conception du modèle MLP :**
   - Définition de l’architecture du réseau.
   - Utilisation de la fonction d’activation ReLU, softmax en sortie.
   - Optimiseur Adam, fonction de perte `categorical_crossentropy`.

3. **Évaluation :**
   - Suivi de la **précision** et de la **perte** sur l’entraînement et la validation.
   - Visualisation des courbes d’apprentissage.
   - Analyse via matrice de confusion.

## **Lancer le notebook**
1. Installer les dépendances :
```bash
pip install -r requirements.txt 
```
2. Ouvrir le fichier main_notebook.ipynb avec Jupyter ou Colab.
3. Explorez le code source.

## **Résultats clés**
   - Le MLP atteint une précision de plus de 80 % sur les mouvements de validation.
   - Les courbes montrent une bonne généralisation, avec un faible surapprentissage.
   - La matrice de confusion indique que les mouvements sont généralement bien distingués, malgré quelques confusions entre blocages et coups similaires.

## **Améliorations futures**
   - Tester des architectures CNN, plus adaptées aux données image.
   - Appliquer davantage d’augmentation de données pour améliorer la robustesse.
   - Étendre la base à d’autres types de katas ou mouvements plus fins.
   - Utiliser d'autres outils afin de traiter des vidéos (PoseNet, OpenPose ou LabelStudio). 

## Remerciements
Merci d’avoir pris le temps de lire cette documentation. J’espère que ce projet vous semblera utile et intéressant !

#### Djibril DAHOUB 
