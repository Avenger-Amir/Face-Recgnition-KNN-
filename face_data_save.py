#In the name of Allah
# Author:-Wajiha

import cv2
import numpy as np

#Init_ Camera
cap=cv2.VideoCapture(0)

#Face Dectection

face_cascade=cv2.CascadeClassifier("C:\\Users\\acer\\Desktop\\ML-Projects\\Face_Recognition\\haarcascade_frontalface_alt.xml")

dataset_path="C:\\Users\\acer\\Desktop\\ML-Projects\\Face_Recognition\\Data\\"
face_data=[]
skip=0;
file_name=input("Enter the name of person: ")

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    #print(faces)
    if len(faces)==0:
        continue
    #print(faces)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    
    #pick the last face face (because it is the largest face according to area (f[2]*f[3]))
    if len(face_data)==10:
        break
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        #Extract (Crop out the requried face): Region of Interest
        
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
            
            
        cv2.imshow("Frame",frame)
        cv2.imshow("Face Section",face_section)
        
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
    
#Convert our face list array into a numpy array

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into file system
np.save(dataset_path+file_name+".npy",face_data)

print("Data Sucessfully saved at"+dataset_path+file_name+".npy")

cap.release();
cv2.destroyAllWindows()
