import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)

count=0
ImageSize=300

folder="D:\PROJECTS\CONTACTLESS LOGIN SYSTEM\Data"
while True:
    sucess, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgCrop=img[y-20:y+h+20,x-20:x+w+20]
        imgWhite = np.ones((ImageSize, ImageSize, 3), np.uint8) * 255

        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = ImageSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, ImageSize))
                wGap = math.ceil((ImageSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = ImageSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (ImageSize, hCal))
                hGap = math.ceil((ImageSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize



            cv2.imshow("CropImage",imgCrop)
            cv2.imshow("WhiteImage",imgWhite)
        except:
            print("Don't come closer")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        count +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(count)