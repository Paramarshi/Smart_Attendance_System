import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

dataset_path = "dataset"
models_path = "models"
model_name = "Facenet"

os.makedirs(models_path, exist_ok=True)

X = []
y = []

print("Extracting embeddings from dataset...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in tqdm(os.listdir(person_folder), desc=f"Processing {person_name}"):
        img_path = os.path.join(person_folder, img_file)

        try:
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False
            )

            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                embedding = embedding_objs[0].get("embedding", None)
                if embedding is not None:
                    X.append(embedding)
                    y.append(person_name)
                else:
                    print(f"No embedding found in {img_path}, skipping.")
            else:
                print(f"No face detected in {img_path}, skipping.")

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total samples collected: {len(X)}")
if len(X) == 0:
    raise ValueError("No embeddings extracted! Check dataset or DeepFace settings.")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Training SVM classifier...")
svm_clf = SVC(kernel="linear", probability=True, class_weight="balanced")
svm_clf.fit(X, y_encoded)

with open(os.path.join(models_path, "attendance_model.pkl"), "wb") as f:
    pickle.dump(svm_clf, f)

with open(os.path.join(models_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete! Models saved in 'models/' folder.")
