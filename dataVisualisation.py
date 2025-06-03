from collections import Counter
import matplotlib.pyplot as plt
import json

def viewData(json_path):

    # Charger les données JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    print("Type de data :", type(data))

    # Afficher un exemple d'entrée
    print(json.dumps(data, indent=4)[:500])

    print(f"Nombre total d'images : {len(data)}")

def viewDataDistribution(json_path):

    # Charger les données JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extraire tous les labels
    labels = [entry["label"] for entry in data.values()]  

    # Compter les occurrences
    label_counts = Counter(labels)

    # Afficher la distribution
    print("Répartition des labels :")
    for label, count in label_counts.items():
        print(f"{label}: {count} images")

    # Plot des classes
    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.keys(), label_counts.values(), color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.title("Répartition des labels dans le dataset")
    plt.xticks(rotation=45)
    plt.show()