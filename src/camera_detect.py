import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os

def detect_facial_emotion():
    os.makedirs('logs/camera_logs', exist_ok=True)
    logging.basicConfig(filename='logs/camera_logs/emotion_detection.log', level=logging.INFO)
    emotion_model = load_model('../pretrained_models/fer2013.hd5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cap = cv2.VideoCapture(0)
    try:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture frame from the camera")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        emotions = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            emotion_pred = emotion_model.predict(face_roi)
            predicted_emotion = emotions_label[np.argmax(emotion_pred)]
            emotions.append(predicted_emotion)
            logging.info(f"Detected emotion: {predicted_emotion}")
            print('Detected Facial Emotion :',predicted_emotion)
        return emotions[0] if emotions else None
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return []
    
    finally:
        cap.release()


if __name__ == "__main__":
    detect_facial_emotion()
# detected_emotion = detect_facial_emotion()
# print('detected_emotion :',detected_emotion)