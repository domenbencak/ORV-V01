import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time

cap = cv2.VideoCapture(0)

while True:
    # Preberi frame iz kamere
    ret, frame = cap.read()
    # Obrni frame
    frame = cv2.flip(frame, 1)
    # Prikaži frame + okvir na regiji
    cv2.imshow('Kamera', frame)

    # Pritisni 'q' za izhod
    if cv2.waitKey(1) == ord('q'):
        break

# Ugasni kamero in uniči vsa okna
cap.release()
cv2.destroyAllWindows()
