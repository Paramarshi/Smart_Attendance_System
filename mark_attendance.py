import os
import pickle
import mysql.connector
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime

MODEL_NAME = "Facenet"
DB_CONFIG = {
    "host": "localhost",
    "user": "root",          # change this
    "password": "password",  # change this
    "database": "attendance_db"
}

with open("models/attendance_model.pkl", "rb") as f:
    svm_clf = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder: LabelEncoder = pickle.load(f)

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL UNIQUE,
    roll_no VARCHAR(20) UNIQUE,
    class VARCHAR(50),
    section VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    date DATE NOT NULL,
    time_in TIME NOT NULL,
    status ENUM('Present','Absent','Late') DEFAULT 'Present',
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    UNIQUE(student_id, date)
)
""")
conn.commit()


def mark_attendance(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist!")
        return

    try:
        faces = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            enforce_detection=False
        )

        if not faces:
            print("No faces detected in the image!")
            return

        for face in faces:
            embedding = np.array(face["embedding"]).reshape(1, -1)
            y_pred = svm_clf.predict(embedding)[0]
            student_name = label_encoder.inverse_transform([y_pred])[0]

            cursor.execute("SELECT student_id FROM students WHERE student_name = %s", (student_name,))
            result = cursor.fetchone()

            if result:
                student_id = result[0]
            else:
                cursor.execute("INSERT INTO students (student_name) VALUES (%s)", (student_name,))
                conn.commit()
                student_id = cursor.lastrowid

            now = datetime.now()
            today = now.date()
            time_in = now.strftime("%H:%M:%S")

            try:
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time_in, status)
                    VALUES (%s, %s, %s, %s)
                """, (student_id, today, time_in, "Present"))
                conn.commit()
                print(f"Marked {student_name} at {time_in}")
            except mysql.connector.errors.IntegrityError:
                print(f"{student_name} already marked for {today}")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    try:
        classroom_photo = input("Enter path to classroom image: ").strip()
        mark_attendance(classroom_photo)
    finally:
        cursor.close()
        conn.close()
