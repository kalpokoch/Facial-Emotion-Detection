import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model



#buttons
def add_todo():
    img_path = filedialog.askopenfilename()
    image_toview =  ctk.CTkImage(Image.open(img_path), size=(250, 250)) 
    print(img_path)

    



    model=load_model(r'C:\Users\lenovo\Documents\project\model_file_30epochs.h5')

    faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    frame = cv2.imread(img_path)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    os.chdir(r'C:\Users\lenovo\Documents\project\Final Working Stable')
    cv2.imwrite('test.png',frame)
    
    image_toview2 =  ctk.CTkImage(Image.open(r"C:\Users\lenovo\Documents\project\Final Working Stable\test.png"), size=(250, 250))

    view_box.configure(image=image_toview2)



def button_event():
     pass
#setting up tkinter
ctk.set_appearance_mode("Dark")
root = ctk.CTk()
root.geometry("750x450")
root.title("Todo App")

#setting up resources
comic_font = ("Eras Light ITC", 20, "bold")

title = ctk.CTkLabel(root, text="Facial Emotion Detector", font=comic_font)
title.pack(padx=10, pady=(40, 20))



frame_a = ctk.CTkFrame(root)
frame_a.pack()

frame_b = ctk.CTkFrame(root)
frame_b.pack()

view_box = ctk.CTkLabel(master=frame_a, text="")
view_box.pack()
add_button = ctk.CTkButton(frame_b, text="Upload Your Image", width=500, command=add_todo)
add_button.pack(pady=20)




root.mainloop()







