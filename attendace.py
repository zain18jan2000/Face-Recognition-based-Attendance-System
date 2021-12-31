import face_recognition
import pickle
import cv2
import os
import pandas as pd
from datetime import datetime
# check if 'record.csv' file exist or not
# if not exist then create one with columns 'NAME' 'DATE' and 'TIME'
if os.path.isfile('record.csv') == False:  
    df ={
    'NAME':[],
    'DATE':[],
    'TIME':[]  
      }
# converting df to dataframe      
    df = pd.DataFrame(df)
    # saving dataframe to csv file
    df.to_csv("record.csv", index = False)
    
def record(name):
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%H:%M:%S")
    data = {
    'NAME': [name],
    'DATE': [date],
    'TIME': [time]
    }
    df = pd.DataFrame(data)
    df.to_csv('record.csv', mode='a', index=False, header=False)

known_names = os.listdir('images')    
# we need to create a dictionary of known people to keep the record
# of persons whose attendace has been marked
conditions = {}
for known_name in known_names:
    conditions[known_name] = False
 
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
 

video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts)
            
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # if a known person found then we will mark his attendance in csv file
            if conditions.get(name,True) == False:
                record(name)
                # Making it sure that attendance is not marked more than once 
                conditions[name] = True
            elif name != "Unknown":
                cv2.putText(frame, 'Attendance Marked', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Attendace App", frame)
    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()