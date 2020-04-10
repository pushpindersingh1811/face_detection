import numpy as np
import cv2
import keras
from keras.models import load_model
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model('model_v6_23.hdf5')

emotion_dict = {'Angry': 0, 'Sad': 5, 'Disgust': 1, 'Neutral': 4, 'Fear': 2, 'Surprise': 6, 'Happy': 3 }

cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.reshape(roi_gray, [1, roi_gray.shape[0], roi_gray.shape[1], 1])
        predicted_class = np.argmax(model.predict(roi_gray))
        label_map = dict((v, k) for k, v in emotion_dict.items())
        predicted_lebal = label_map[predicted_class]
        cv2.putText(frame, 'Status = {}'.format(predicted_lebal), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()