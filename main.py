from tkinter import *
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

inputs = []
prev=[0]
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier=Classifier("Model/keras_model.h5","Model/labels.txt")
ImageSize=300
labels=["0","1","2","3","4","5","6","7","8","9","BACK","ENTER","NEXT","SUBMIT"]
got=[]

def Main():

    def login(username,password):

        if (username == "" and password == ""):
            messagebox.showinfo("", "Blank Not allowed")

        elif (username == "00" and password == "00"):
            messagebox.showinfo("", "login success")

        else:
            messagebox.showinfo("", "incorrect username or password")



    root = Tk()
    root.title("Login")
    root.geometry("450x500")

    global entry1
    global entry2


    label_1 = Label(root)


    def to_pil(img, label, x, y, w, h):
        img = cv2.resize(img, (w, h))
        image = Image.fromarray(img)
        pic = ImageTk.PhotoImage(image)
        label.configure(image=pic)
        label.image = pic
        label.place(x=x, y=y)

    def display():
            sucess, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]
                imgWhite = np.ones((ImageSize, ImageSize, 3), np.uint8) * 255

                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        k = ImageSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, ImageSize))
                        wGap = math.ceil((ImageSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)

                    else:
                        k = ImageSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (ImageSize, hCal))
                        hGap = math.ceil((ImageSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)

                    inputstring = ''.join(inputs)
                    if labels[index] == "ENTER":
                        inputs.append(labels[prev[0]])

                    elif labels[index]=="BACK":
                        inputs.pop()

                    elif labels[index]=="NEXT":
                        if inputstring!="":
                            got.append(inputstring)
                            inputs.clear()
                            print(got)
                    elif labels[index]=="SUBMIT":
                        a=got[0]
                        b=got[1]
                        got.clear()
                        login(a,b)

                    else:
                        prev.pop()
                        prev.append(index)
                    print(inputs)
                    print(got)
                    print("------------------")


                    cv2.rectangle(imgOutput, (x - 20, y - 70), (x + 120, y-20), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255),2)
                    cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 255), 4)
                    cv2.putText(imgOutput, inputstring, (10,460), cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 0, 255),4)
                except:
                    print("Alert", "Show hand clearly")


            cv2.waitKey(1000)
            finalimage = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
            to_pil(finalimage, label_1, 10, 10, 400, 300)
            label_1.after(20, display)


    display()


    Label(root, text="Username").place(x=20, y=350)
    Label(root, text="Password").place(x=20, y=370)

    entry1 = Entry(root, bd=5)
    entry1.place(x=140, y=350)

    entry2 = Entry(root, bd=5)
    entry2.place(x=140, y=370)

    Button(root, text="Login", command=login, height=3, width=13, bd=6).place(x=100, y=400)


    root.mainloop()

Main()