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
    roi_image = slika[int(levo_zgoraj[1]):int(levo_zgoraj[1] + desno_spodaj[1]),
                int(levo_zgoraj[0]):int(levo_zgoraj[0] + desno_spodaj[0])]
    avg_color = np.int_(np.average(roi_image, axis=(0, 1)))
    std_dev = np.int_(np.std(roi_image))
    return avg_color - std_dev, avg_color + std_dev

def prestej_piksle_z_barvo_koze(podslika, barva_koze_spodaj, barva_koze_zgoraj):
    width, height, channel = np.int_(podslika.shape)
    count = 0
    for row in range(podslika.shape[0]):
        for col in range(podslika.shape[1]):
            pixel = podslika[row, col]
            if (barva_koze_spodaj <= pixel).all() and (pixel <= barva_koze_zgoraj).all():
                count += 1
                # print("count: ", count)
    return count

def obdelaj_sliko(slika, okno_sirina, okno_visina, barva_koze_spodaj, barva_koze_zgoraj):
    najboljse_ujemanje = -1
    # Nastavi (korak) na 20% širine in višine
    visina = int(okno_visina * 0.2)
    sirina = int(okno_sirina * 0.2)
    for y in range(0, int(okno_visina), visina):
        for x in range(0, int(okno_sirina), sirina):
            # Izreži trenutno okno
            okno = slika[y:y + visina, x:x + sirina]
            top_left = (x, y)
            bottom_right = (x + sirina, y + visina)
            # Dobi trenutno število pikslov, ki se ujemajo
            trenutno_ujemanje = prestej_piksle_z_barvo_koze(okno, barva_koze_spodaj, barva_koze_zgoraj)
            # Preveri ali si prvič v zanki in nato shrani v najboljše ujemanje prvi okvirček
            if najboljse_ujemanje == -1:
                najboljse_ujemanje = trenutno_ujemanje
                najboljse_tl = top_left
                najboljse_br = bottom_right
            # Če je trenutno ujemanje boljše od najboljšega do zdaj
            elif trenutno_ujemanje > najboljse_ujemanje:
                najboljse_ujemanje = trenutno_ujemanje
                najboljse_tl = top_left
                najboljse_br = bottom_right

    return (najboljse_tl, najboljse_br)


# Začni zajemanje videa, počakaj 1 sekundo (da senzor dobi dovolj svetlobe) nato zajemi prvi frame
cap = cv2.VideoCapture(0)
time.sleep(1)
_, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = zmanjsaj_sliko(frame)
# Zberi ROI od levega kota zgoraj do desnega kota spodaj
roi = cv2.selectROI(frame, fromCenter=False)
#roi_image = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
#print("ROI: ", (roi[0], roi[1]), (roi[2], roi[3]))
# Določi barvo kože, ki predstavlja obraz, ki ga iščemo
levo_zgoraj = (roi[0], roi[1])
desno_spodaj = (roi[2], roi[3])
bk_spodaj, bk_zgoraj = doloci_barvo_koze(frame, levo_zgoraj, desno_spodaj)
# Zapri vsa okna in začni nov capture
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

while True:
    # Preberi frame iz kamere
    ret, frame = cap.read()
    # Obrni frame
    frame = cv2.flip(frame, 1)
    # Zmanjšaj sliko
    frame = zmanjsaj_sliko(frame)
    # Pridobi visino in sirino okna (slike)
    visina, sirina, channel = np.int_(frame.shape)
    # Obdelaj sliko in shrani levi zgornji kot in desni spodnji kot regije, ki se najbolše ujema z izbranim
    topleft, bottomright = obdelaj_sliko(frame, sirina, visina, bk_spodaj, bk_zgoraj)
    # Nariši okvir na to regijo
    cv2.rectangle(frame, topleft, bottomright, (0, 255, 0), 2)
    # Prikaži frame + okvir na regiji
    cv2.imshow('Kamera', frame)

    # Pritisni 'q' za izhod
    if cv2.waitKey(1) == ord('q'):
        break

# Ugasni kamero in uniči vsa okna
cap.release()
cv2.destroyAllWindows()