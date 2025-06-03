import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer , scale, LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns


datapath = 'dataFirstSet'
model = 'modelMLPV1'
checkpoint_filepath = 'models/best_modelV1.keras'


with open(datapath, "r") as f:
    data = json.load(f)

all_features = []
all_labels = []

for image_name, infos in data.items():
    landmarks = infos["landmarks"]
    label = infos["label"]

    all_features.append(np.array(landmarks).flatten()) # transform to a 1D vector
    all_labels.append(label)

train_features, test_features, train_label, test_label = train_test_split(
    all_features,all_labels,test_size=0.3,random_state=42,stratify=all_labels
)

lb = LabelEncoder()
y_train = lb.fit_transform(train_label)
y_test = lb.transform(test_label)

X_train = np.array(train_features)
X_test = np.array(test_features)

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

#Building the MLP model =>
model = Sequential([
    Dense(64, activation='relu',input_shape=(132,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model compilation =>
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resume
model.summary

# Keep the best model
tf.keras.callbacks.ModelCheckpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test,y_test)
print(f"Précisions : {test_acc:.4f}")

# Courbe de la perte et de la précision
plt.figure(figsize=(12, 5))

# Courbe de la perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Épochs")
plt.ylabel("Perte")
plt.legend()
plt.title("Évolution de la perte")

# Courbe de la précision
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Épochs")
plt.ylabel("Précision")
plt.legend()
plt.title("Évolution de la précision")

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
# confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix with True Class Names")

plt.show()