#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import face_recognition
import numpy as np
import os
from glob import glob


# In[2]:


class Facerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names= []
        
        #resize frames for faster speed
        self.frame_resizing= 0.25
        
    def load_encoding_image(self,image_path):
        """
        Load encoding images from path
        params: image_path
        return:
        """
        
        #Load image
        image_path = glob(os.path.join(image_path , "*.*"))
        
        print("{} encoding images found.".format(len(image_path)))
        
        #Store image encoding and names
        for img_path in image_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            
            #Get the file name from the initial file path
            basename = os.path.basename(img_path)
            (filename , ext) = os.path.splitext(basename)
            #Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            
            #Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            
        print("Image enconding has loaded")
        
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame ,(0,0), fx= self.frame_resizing , fy= self.frame_resizing)
        # Find all the faces and face encoding in the current frames of video
        #Convert the image from BGR color to RGB color 
        rgb_small_frame= cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_names= []
        for fcae_encoding in face_encodings:
            #Check if the face is a match for the known faces
            matches= face_recognition.compare_faces(self.known_face_encodings, fcae_encoding)
            name= "Unknown"
            
            face_distances = face_recognition.face_distance(self.known_face_encodings , fcae_encoding)
            best_match_index= np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                
            face_names.append(name)
            
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int) , face_names

