import cv2
import numpy as np
import tkinter as tk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Two classes: right and left
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Detect face and eyes
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    return eyes, roi_color if len(faces) > 0 else None

# Extract features from eye regions
def extract_eye_features(eyes, frame):
    features = []
    for (ex, ey, ew, eh) in eyes:
        eye_frame = frame[ey:ey+eh, ex:ex+ew]
        gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        resized_eye = cv2.resize(gray_eye, (64, 64))
        features.append(resized_eye.reshape(64, 64, 1))
    return np.array(features)

# Real-time interpretation of eye movements
def interpret_eye_movements(frame, model):
    eyes, roi_color = detect_eyes(frame)
    if len(eyes) > 0:
        features = extract_eye_features(eyes, roi_color)
        predictions = model.predict(features)
        for prediction in predictions:
            if np.argmax(prediction) == 1:
                print("Patient needs food")
            elif np.argmax(prediction) == 0:
                print("Patient needs water")

# Create a simple UI to show the system is monitoring
def create_interface():
    root = tk.Tk()
    label = tk.Label(root, text="Monitoring Patient...")
    label.pack()
    root.mainloop()

# Main function to run the detection system
def main():
    model = build_model()
    
    # Load or train the model here
    # For demonstration, let's assume we have a pre-trained model
    # model.load_weights('path_to_pretrained_model.h5')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        interpret_eye_movements(frame, model)
        
        cv2.imshow('Eye Movement Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the main function in a separate thread to keep the UI responsive
if __name__ == "__main__":
    import threading
    threading.Thread(target=create_interface).start()
    main()
