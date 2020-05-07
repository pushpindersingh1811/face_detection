import cv2
import face_recognition
import os
import numpy as np


FACES_FOLDER = 'faces'
MODEL = 'hog'

cap = cv2.VideoCapture(0)

def get_encoded_faces():

    encoded = {}

    for f in os.listdir(FACES_FOLDER):
        if f.endswith('.jpg') or f.endswith('.png'):
            face = face_recognition.load_image_file(os.path.join(FACES_FOLDER, f))
            encoding = face_recognition.face_encodings(face)[0]
            encoded[f.split('.')[0]] = encoding

    return encoded



def classify_face(img):

    faces = get_encoded_faces()
    known_faces = list(faces.values())
    known_names = list(faces.keys())

    #img = cv2.imread(im, 1)


    face_locations = face_recognition.face_locations(img, model=MODEL)
    unkown_image_encoding = face_recognition.face_encodings(img, face_locations)

    face_name = []

    for face_encoding in unkown_image_encoding:
        results = face_recognition.compare_faces(known_faces, face_encoding)
        name = 'unkown'

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if results[best_match_index]:
            name = known_names[best_match_index]

        face_name.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_name):
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (0, 0, 255), 2)

            cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (left-20, bottom+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #while True:

        #cv2.imshow('Image', img)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #return face_name

while cap.isOpened():

    ret, img = cap.read()

    classify_face(img)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
