import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time

def zmanjsaj_sliko(slika):
    # Vrni zmanjšano sliko
    return cv2.resize(slika, (300, 260))

# Začni zajemanje videa, počakaj 1 sekundo (da senzor dobi dovolj svetlobe) nato zajemi prvi frame
cap = cv2.VideoCapture(0)
time.sleep(1)
_, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = zmanjsaj_sliko(frame)
# Zberi ROI od levega kota zgoraj do desnega kota spodaj
roi = cv2.selectROI(frame, fromCenter=False)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

while True:
    # Preberi frame iz kamere
    ret, frame = cap.read()
    frame = zmanjsaj_sliko(frame)
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
