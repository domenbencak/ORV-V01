import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time

def zmanjsaj_sliko(slika):
    # Vrni zmanjšano sliko
    return cv2.resize(slika, (300, 260))

def doloci_barvo_koze(slika, levo_zgoraj, desno_spodaj):
    # Določi povprečje in standardno deviacijo pikslov izbrane regije na sliki
    roi_image = slika[int(levo_zgoraj[1]):int(levo_zgoraj[1] + desno_spodaj[1]), int(levo_zgoraj[0]):int(levo_zgoraj[0] + desno_spodaj[0])]
    avg_color = np.int_(np.average(roi_image, axis=(0, 1)))
    std_dev = np.int_(np.std(roi_image))
    return avg_color - std_dev, avg_color + std_dev

# Začni zajemanje videa, počakaj 1 sekundo (da senzor dobi dovolj svetlobe) nato zajemi prvi frame
cap = cv2.VideoCapture(0)
time.sleep(1)
_, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = zmanjsaj_sliko(frame)
# Zberi ROI od levega kota zgoraj do desnega kota spodaj
roi = cv2.selectROI(frame, fromCenter=False)
levo_zgoraj = (roi[0], roi[1])
desno_spodaj = (roi[2], roi[3])
bk_spodaj, bk_zgoraj = doloci_barvo_koze(frame, levo_zgoraj, desno_spodaj)
print("s, z: ", bk_spodaj, bk_zgoraj)
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
