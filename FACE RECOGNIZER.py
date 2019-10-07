#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
    face detection and recognition using opencv
'''

train0 = 'H:/python/all1/Bill gates'
train1 = 'H:/python/all1/elon musk'
algopath = 'H:/python/all1/haarcascade_frontalface_default.xml'


# In[2]:


import cv2
import os
import numpy as np
from tqdm import tqdm
import time



# In[3]:

#adding the classifier file 
fcascade = cv2.CascadeClassifier(algopath)


# In[4]:


train_data = []
labels =[]
i=0
def training_data(directory,label):
    for x in tqdm(os.listdir(directory)):
        path = os.path.join(directory , x)
        gray_img = cv2.imread(path ,0)

        if gray_img is None:
            continue
        gray_img = cv2.resize(gray_img, (500,500))
        faces_rect =fcascade.detectMultiScale(gray_img,scaleFactor=1.10,minNeighbors = 1)
        if len(faces_rect)==0:
            continue
        (x,y,w,h) = faces_rect[0]
        grayfaceonly = gray_img[y:y+h,x:x+w]
        train_data.append(grayfaceonly) 
        labels.append(label)
    return train_data,labels


# In[15]:

#labelling the data
t_data0 , label0 = training_data(train0,[0])
t_data1 , label1 = training_data(train1,[1])


# In[16]:


t_data = t_data0 + t_data1
t_labels = label0 + label1


# In[17]:


face_recg = cv2.face.LBPHFaceRecognizer_create()


# In[18]:


face_recg.train(t_data , np.array(t_labels))


# In[24]:


test_img = cv2.imread('H:/python/all1/elon-musk-and-bill-gates.jpg')


# In[25]:


test_img = cv2.resize(test_img,(int(test_img.shape[1]/1.2),int(test_img.shape[0]/1.2)))
gray_img = cv2.cvtColor(test_img , cv2.COLOR_RGB2GRAY)


# In[26]:


faces_rect =fcascade.detectMultiScale(gray_img,scaleFactor=1.1,minSize=(25,25),minNeighbors = 5,flags = cv2.CASCADE_SCALE_IMAGE)


# In[27]:


name = { 0 : "Bill Gates" , 1 : "Elon Musk"}


# In[28]:


for (x,y,w,h) in faces_rect:
    value , conf = face_recg.predict(gray_img[y:y+h,x:x+w])
    if conf>40 and conf<100:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(145,14,15),2)
        cv2.putText(test_img,name[value],(x,y-5), cv2.FONT_HERSHEY_DUPLEX,1,(2,152,10),1)
        cv2.imshow("FACE RECOGNIZER",test_img)

    if conf > 100:
        continue
    # cv2.rectangle(test_img,(x,y),(x+w,y+h),(145,14,15),2)
    # cv2.putText(test_img,name[value],(x,y-5), cv2.FONT_HERSHEY_DUPLEX,1,(2,152,10),1)
    # cv2.imshow("FACE RECOGNIZER",test_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




