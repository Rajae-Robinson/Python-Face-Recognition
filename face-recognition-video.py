import cv2 # opencv library
from random import randrange

# load pre-trained data on face frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capturing video from webcam
# VideoCapture(0) uses default webcam
webcam = cv2.VideoCapture('funny.mp4')

# Iterate forever over frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()

    # converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # drawing rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # Showing the image
    cv2.imshow('Matter Industries Face Detector', frame)
    # displaying one frame every milisecond
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the VideoCapture object
webcam.release()

print('Code completed')