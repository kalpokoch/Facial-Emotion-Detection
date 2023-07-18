import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import load_model

model=load_model(r'C:\Users\lenovo\Documents\python practise\best.h5')

faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
frame = cv2.imread(r"C:\Users\lenovo\Documents\project\WhatsApp Image 2023-07-17 at 10.57.28 AM.jpeg")
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(gray, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 48,48, 1))
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

os.chdir(r'C:\Users\lenovo\Documents\project')
cv2.imwrite('test.jpg',frame)
img = mpimg.imread('test.jpg')
imgplot = plt.imshow(img)

cv2.imshow("Frame",frame)
k=cv2.waitKey(0)
