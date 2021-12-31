# Face-Recognition-based-Attendance-System
A real time implementation of Attendance System in python.

<h1>Pre-requisites</h1>
To understand the implentation of Facial recognition based Attendance System you  must have, <br>
– Basic understanding of Image Classification<br>
– Knowledge of Python and Deep Learning<br>
<br>
<h1>Dependencies</h1>
1- OpenCV <br>
2- dlib<br>
3- face_recognition <br>
4- os <br>
5- imutils <br>
6- numpy <br>
7- pickle <br>
8- datetime <br>
9- Pandas <br><br>

<b>Note:</b> To install dlib and face_recognition, you need to create a virtual environment in your IDE first.<br>

<h1>Overview</h1>
Face is the crucial part of the human body that uniquely identifies a person. Using the face characteristics the face recognition projects can be implemented. The  technique which I have used  to  implent this priject is <b>Deep  Metric Learning</b>. <br> 
  
<h1>What is Deep  Metric Learning ?</h1> 
If you have any prior experience with deep learning you know that we typically train a network to: <br>
Accept a single input image<br>
And output a classification/label for that image <br>
However, deep metric learning is different. Instead, of trying to output a single label, we are outputting a real-valued feature vector.  This technique can be divided into three steps,<br>
  
  <h2>Face Detection</h2>  
  The first task that we perform is detecting faces in the image(photograph) or video stream. Now we know that the exact coordinates or location of the face, so we extract this face for further processing.<br>
  
  <h2>Feature Extraction</h2>
  Now see we have cropped out the face from the image, so we extract specific features from it. Here we are going to see how to use face embeddings to extract these features of the face. Here  a neural network takes an image of the face of the person as input and outputs a vector that represents the most important features of a face. For the dlib facial recognition network which I have used here, the output feature vector is 128-d (i.e., a list of 128 real-valued numbers) that is used to quantify the face. This output feature vector is also called face embeddings.<br>
  
  <h2>Comparing Faces</h2>  
  We have face embeddings for each face in our data saved in a file, the next step is to recognize a new image that is not in our data. Hence the first step is to compute the face embedding for the image using the same network we used earlier and then compare this embedding with the rest of the embeddings that we have. We recognize the face if the generated embedding is closer or similar to any other embedding.<br>


  <h1>What's include in this repository ?</h1>
  Three files only. These are <br>
  1- <b>Feature_extractor.py</b> for extracting and saving the features from images (128-d vector for each face) of persons provided<br>
  2- <b>attendace.py</b> for real time implementation of Face-Recognition-based-Attendance-System. <br>
  3- <b>README.md</b> which you are reading.<br>
  
  <h1>How to implement ?</h1>
  Create a folder named 'images' at the same location where you have kept the python files mentioned above. In this folder you will create sub folders, each having the the images of the of a persons whom you want the program to recognize. Each folder should have 3-4 images. Change the names of subfolder to the names of the people to be identified.
  Now first run <b>Feature_extractor.py</b>. This will takes some time and provide you a file named <b> face_enc </b> containing the features extracted from the the images. This file will be used by <b>attendace.py</b>. Now run attendace.py to real time implementation of Face-Recognition-based-Attendance-System. If a person recognized by the program then his/her name and time of recognization will be stored in <b> record.csv </b>. You don't need this file, program will create this file itself and will keep maintaining the attendance data in it. <br>
  To further understand the working of program, just go through the code files and read the comments in it.
  


