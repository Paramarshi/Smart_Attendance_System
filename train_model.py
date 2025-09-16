import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
dataset_path = "dataset"
models_path = "models"
model_name = "Facenet"
embedding_size = 128 if model_name == "Facenet" else 2622

# -----------------------------
# CREATE MODELS FOLDER IF NOT EXISTS
# -----------------------------
if not os.path.exists(models_path):
    os.makedirs(models_path)

# -----------------------------
# INITIALIZE DATA LISTS
# -----------------------------
X = []
y = []

print("Extracting embeddings from dataset...")

# -----------------------------
# LOOP THROUGH DATASET
# -----------------------------
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in tqdm(os.listdir(person_folder), desc=f"Processing {person_name}"):
        img_path = os.path.join(person_folder, img_file)

        try:
            # Extract embedding
            embedding_obj = DeepFace.represent(
                img_path=img_path, model_name=model_name, enforce_detection=False
            )

            if embedding_obj and "embedding" in embedding_obj[0]:
                embedding = embedding_obj[0]["embedding"]
                X.append(embedding)
                y.append(person_name)
            else:
                print(f"No face detected in {img_path}, skipping.")

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

# -----------------------------
# CONVERT TO NUMPY ARRAYS
# -----------------------------
X = np.array(X)
y = np.array(y)

print(f"Total samples collected: {len(X)}")

# -----------------------------
# ENCODE LABELS
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# TRAIN SVM CLASSIFIER
# -----------------------------
print("Training SVM classifier...")
svm_clf = SVC(kernel="linear", probability=True)
svm_clf.fit(X, y_encoded)

# -----------------------------
# SAVE MODEL AND ENCODER
# -----------------------------
with open(os.path.join(models_path, "attendance_model.pkl"), "wb") as f:
    pickle.dump(svm_clf, f)

with open(os.path.join(models_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete! Models saved in 'models/' folder.")
