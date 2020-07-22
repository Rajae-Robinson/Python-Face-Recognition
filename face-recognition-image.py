import cv2 # opencv library

"""
    This program uses Computer Vison, a subset of AI
    
    The aim is the train the machine to recognize a face.
    In order to do that you need to use a lot of faces as input data
    to the algorithm. It then analyzes the images and pick up patterns
    on what makes a face look like a face and what a face doesn't look like

    In this program, data which is already trained is used. It is done
    by using the Haar Cascade Algorithm, which detects faces by the difference
    in brightness of adjacent pixels. Hence, the images are converted to greyscale

"""

# load pre-trained data on face frontals
# Cascade - name of algorithm
# Classifier - another word for detector
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('avengers.png')

# converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


# Drawing rectangle around faces
# face_coordinates returns a list of lists where
# the sublists contain a top left x and y position for the starting point of the
# shape and width and height which can be added to the top left x, y points to get
# the bottom right position of the rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Showing the image
cv2.imshow('Matter Industries Face Detector', img)
# waitkey is needed to show the image
cv2.waitKey()

print('Code completed')