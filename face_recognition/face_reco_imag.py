import face_recognition
import cv2
import os

KNOWN_IMAGE_DIR = 'Known_Images'
UNKNOWN_IMAGE_DIR = 'Unknown_Images'
TOLERANCE = 0.4
MODEL = 'hog'

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_IMAGE_DIR):
    for filename in os.listdir(f'{KNOWN_IMAGE_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_IMAGE_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('Preprocessing the faces...')

for filename in os.listdir(UNKNOWN_IMAGE_DIR):
    print(f'Filename>> {filename}')

    image = face_recognition.load_image_file(f'{UNKNOWN_IMAGE_DIR}/{filename}')
    location = face_recognition.face_locations(image, model=MODEL)
    encoding = face_recognition.face_encodings(image, location)
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding, location):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 255, 255))

        cv2.imshow(filename, image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()