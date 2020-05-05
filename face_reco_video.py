import face_recognition
import cv2
import os

KNOWN_IMAGE_DIR = 'Known_Images'
#UNKOWN_IMAGE_DIR = 'Unknown_Images'
TOLERANCE = 0.6
MODEL = 'hog'

cap = cv2.VideoCapture(0)

print('Loading known faces...')

known_lebals = []
known_images = []

for name in os.listdir(KNOWN_IMAGE_DIR):
    for filename in os.listdir(f'{KNOWN_IMAGE_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_IMAGE_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)
        known_images.append(encoding)
        known_lebals.append(name)

#print('Preprocessing unknown faces...')

while True:
    #print(filename)
    #image = face_recognition.load_image_file(f'{UNKOWN_IMAGE_DIR}/{filename}')
    ret, image = cap.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    encoding = face_recognition.face_encodings(image, locations)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding, locations):
        results = face_recognition.compare_faces(known_images, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_lebals[results.index(True)]
            print(f'Match Found: {match}')
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

