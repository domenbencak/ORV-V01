# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# GitHub push problem solved by: https://www.codingem.com/git-fix-support-for-password-authentication-was-removed-error/

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

cap = cv2.VideoCapture(1)

if cap.isOpened() == False:
    print("Ne morem odpreti kamere")

cv2.namedWindow("Kamera")
while True:
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        cv2.imshow("Kamera", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
