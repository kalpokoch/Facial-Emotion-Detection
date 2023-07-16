import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

model = load_model(r'C:\Users\lenovo\Documents\project\Final Working Stable\best_weights.h5')
faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
frame = cv2.imread(r'C:\Users\lenovo\Documents\project\Final Working Stable\Screenshot (85).png')
gray = frame
faces = faceDetect.detectMultiScale(frame, 1.3, 3)
for x,y,w,h in faces:
    sub_faces = gray[y:y+h, x:x+w]
    resize = cv2.resize(sub_faces,(256,256))
    normalize = resize/255.0
    reshaped = np.reshape(normalize, (1,256,256,3)) 
    result = model.predict(reshaped)
    label = np.argmax(result, axis = 1)[0]
    print(label)
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,256), 3)

    cv2.putText(frame, label_dict[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,256), 2)

os.chdir(r'C:\Users\lenovo\Documents\project')
cv2.imwrite('test.jpg',frame)
img = mpimg.imread('test.jpg')
imgplot = plt.imshow(img)

cv2.imshow("Frame",frame)
k=cv2.waitKey(0)