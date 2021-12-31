import face_recognition
import pickle
import cv2
import os
from imutils import paths
import numpy as np

# Extracting Features
knownEncodings = list()
knownNames = list()
encodings = []
imagePaths = list(paths.list_images('images'))

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    if boxes == []:
        print('Face not detected')
        cv2.imshow('Face not dtected in this pic. Please replace it',rgb)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
        break
    # find out the encodings (a vector of 128 measurements)    
    encodings = face_recognition.face_encodings(rgb, boxes)
    knownEncodings.append(encodings[0])
    knownNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()