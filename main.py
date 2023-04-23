import time
import cv2
import numpy as np
import scipy
import math
import random


def kmeans(slika, centri, iteracije):
    st_centrov = len(centri)
    novi_centri = centri
    # Zanka za vse iteracije
    for it in range(iteracije):
        groups = [[] for _ in range(st_centrov)]
        array_for_reconstruction = []
        # Pojdi skozi vsak pixel slike
        for row in range(slika.shape[0]):
            for col in range(slika.shape[1]):
                # Pridobi barve
                r1 = slika[row, col, 2]
                g1 = slika[row, col, 1]
                b1 = slika[row, col, 0]
                # Shrani v pixel barve, če je RGB nacin, drugače shrani v pixel xyRGB
                pixel = [r1, g1, b1]
                if len(centri[0]) == 5:
                    pixel = [row, col, r1, g1, b1]
                # Izračunaj vse razdalje piksla do vsakega centra
                razdalje = [evklidska_razdalja(pixel, novi_centri[i]) for i in range(st_centrov)]
                # Pridobi indeks najmanjše razdalje
                min_index = razdalje.index(min(razdalje))
                # Dodaj pixel v array groups na index minimalne razdalje
                groups[min_index].append(pixel)
                # V pixel dodaj na 4. / 6. mesto minimalni index razdalje
                pixel_with_index = pixel + [min_index]
                # Dodaj pixel z novim atributom v nov array, ki se kasneje uporabi za rekonstrukcijo
                array_for_reconstruction.append(pixel_with_index)

        # Posodobi nove centre na povprečje vsake skupine
        novi_centri = [np.mean(groups[i], axis=0).astype(int) for i in range(st_centrov)]

    # Pridobi barve iz novih centrov in jih shrani v array
    colors = []
    if len(novi_centri[0]) == 5:
        for i in range(len(novi_centri)):
            colors.append((novi_centri[i][4], novi_centri[i][3], novi_centri[i][2]))
    else:
        for i in range(len(novi_centri)):
            colors.append((novi_centri[i][2], novi_centri[i][1], novi_centri[i][0]))

    # Rekonstrukcija slike
    segmentirana_slika = np.zeros(slika.shape, dtype=np.uint8)
    for row in range(slika.shape[0]):
        for col in range(slika.shape[1]):
            # Izračunaj linearen indeks prave pozicije v sliki
            # (namesto 2D arraya kot je slika imam 1D array, v katerem so vsi piksli in indeks njihove skupine)
            index = row * slika.shape[1] + col
            new_pixel = array_for_reconstruction[index]
            # Če je xyRGB način, pridobi index barve s šestega mesta
            if len(centri[0]) == 5:
                color = colors[new_pixel[5]]
            # Če je pa samo RGB način, pridobi index barve s četrtega mesta
            elif len(centri[0]) == 3:
                color = colors[new_pixel[3]]
            # V trenutno mesto na sliki (x, y) shrani dobljeno barvo
            segmentirana_slika[row, col] = color

    # Pokaži segmentirano sliko z unikatnim idjem
    cv2.imshow(f"{len(colors)} barv in {iteracije} iteracij (id: {random.randint(0, 9999)})", segmentirana_slika)
    '''
    print("Centri na koncu:")
    print(novi_centri)
    '''


def kmeans_original_colors(slika, centri, iteracije):
    st_centrov = len(centri)
    novi_centri = centri
    # Pridobi barve iz centrov in jih shrani v array
    colors = []
    if len(novi_centri[0]) == 5:
        for i in range(len(novi_centri)):
            colors.append((novi_centri[i][4], novi_centri[i][3], novi_centri[i][2]))
    else:
        for i in range(len(novi_centri)):
            colors.append((novi_centri[i][2], novi_centri[i][1], novi_centri[i][0]))
    # Zanka za vse iteracije
    for it in range(iteracije):
        groups = [[] for _ in range(st_centrov)]
        array_for_reconstruction = []
        # Pojdi skozi vsak pixel slike
        for row in range(slika.shape[0]):
            for col in range(slika.shape[1]):
                # Pridobi barve
                r1 = slika[row, col, 2]
                g1 = slika[row, col, 1]
                b1 = slika[row, col, 0]
                # Shrani v pixel barve, če je RGB nacin, drugače shrani v pixel xyRGB
                pixel = [r1, g1, b1]
                if len(centri[0]) == 5:
                    pixel = [row, col, r1, g1, b1]
                # Izračunaj vse razdalje piksla do vsakega centra
                razdalje = [evklidska_razdalja(pixel, novi_centri[i]) for i in range(st_centrov)]
                # Pridobi indeks najmanjše razdalje
                min_index = razdalje.index(min(razdalje))
                # Dodaj pixel v array groups na index minimalne razdalje
                groups[min_index].append(pixel)
                # V pixel dodaj na 4. / 6. mesto minimalni index razdalje
                pixel_with_index = pixel + [min_index]
                # Dodaj pixel z novim atributom v nov array, ki se kasneje uporabi za rekonstrukcijo
                array_for_reconstruction.append(pixel_with_index)

        # Posodobi nove centre na povprečje vsake skupine
        novi_centri = [np.mean(groups[i], axis=0).astype(int) for i in range(st_centrov)]

    # Rekonstrukcija slike
    segmentirana_slika = np.zeros(slika.shape, dtype=np.uint8)
    for row in range(slika.shape[0]):
        for col in range(slika.shape[1]):
            # Izračunaj linearen indeks prave pozicije v sliki
            # (namesto 2D arraya kot je slika imam 1D array, v katerem so vsi piksli in indeks njihove skupine)
            index = row * slika.shape[1] + col
            new_pixel = array_for_reconstruction[index]
            # Če je xyRGB način, pridobi index barve s šestega mesta
            if len(centri[0]) == 5:
                color = colors[new_pixel[5]]
            # Če je pa samo RGB način, pridobi index barve s četrtega mesta
            elif len(centri[0]) == 3:
                color = colors[new_pixel[3]]
            # V trenutno mesto na sliki (x, y) shrani dobljeno barvo
            segmentirana_slika[row, col] = color

    # Pokaži segmentirano sliko z unikatnim idjem
    cv2.imshow(f"{len(colors)} barv in {iteracije} iteracij (id: {random.randint(0, 9999)})", segmentirana_slika)
    '''
    print("Centri na koncu:")
    print(novi_centri)
    '''


def evklidska_razdalja(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same number of coordinates")
    point1 = np.array(point1, dtype=np.int32)
    point2 = np.array(point2, dtype=np.int32)
    squared_distance = 0
    for i in range(len(point1)):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)


def izracunaj_centre(slika, nacin, dimenzije):
    if nacin == "rocno":
        cv2.imshow("Select Centers", slika)
        centri = []

        def on_mouse(event, x, y, flags, params):
            '''
            if len(centri) >= 3:
                return centri
            '''
            if event == cv2.EVENT_LBUTTONUP:
                if x < slika.shape[1] and y < slika.shape[0]:
                    # Če je izbran način xyRGB dodaj v centre xyRGB koordinate
                    if dimenzije == "xyRGB":
                        centri.append((x, y, slika[y, x, 2], slika[y, x, 1], slika[y, x, 0]))
                    # Če je pa izbran način RGB pa dodaj le RGB koordinate
                    elif dimenzije == "RGB":
                        centri.append((slika[y, x, 2], slika[y, x, 1], slika[y, x, 0]))
                    # Nariši moder krog na kliknjeno mesto
                    #cv2.circle(slika, (x, y), 3, (255, 0, 0), -1)
                    cv2.imshow("Select Centers", slika)

        cv2.setMouseCallback("Select Centers", on_mouse)
        # Ob kliku katere koli tipke (potrditev) nadaljuj (vrni centre)
        cv2.waitKey(0)
        cv2.destroyWindow("Select Centers")
        print("Centers: ")
        print(centri)
        return centri
    elif nacin == "nakljucno":
        centri = []
        width, height, _ = slika.shape
        min_distance = 50
        while True:
            # Izberi naključno x in y koordinato ter pridobi njune barve
            x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
            r1 = slika[x1, y1, 2]
            g1 = slika[x1, y1, 1]
            b1 = slika[x1, y1, 0]
            # Če so centri še prazni, dodaj center
            if not centri:
                if dimenzije == "RGB":
                    centri.append((r1, g1, b1))
                else:
                    centri.append((x1, y1, r1, g1, b1))
                continue

            for i in range(len(centri)):
                if dimenzije == "RGB":
                    center = (r1, g1, b1)
                    print(f"Centri[{i}]:")
                    print(centri[i])
                    # Izračunaj razdaljo med centri
                    razdalja = evklidska_razdalja(center, centri[i])
                else:
                    center = [x1, y1, r1, g1, b1]
                    razdalja = evklidska_razdalja(center, centri[i])
                # Če je razdalja premajhna, ponovi postopek
                if razdalja < min_distance:
                    break
            # Če so vsi pogoji OK, dodaj centre v array
            else:
                if dimenzije == "RGB":
                    centri.append((r1, g1, b1))
                else:
                    centri.append((x1, y1, r1, g1, b1))
            # Nastavi število centrov
            if len(centri) == 12:
                break
    print("Centri: ")
    print(centri)
    return centri


slika = cv2.imread('ples_small.jpeg')
cv2.imshow('Original Zelenjava', slika)

centri = izracunaj_centre(slika, "nakljucno", "RGB")

'''
kmeans(slika, centri, 1)
kmeans(slika, centri, 2)
kmeans(slika, centri, 3)
kmeans(slika, centri, 5)
kmeans(slika, centri, 10)
kmeans(slika, centri, 20)

start = time.perf_counter()
kmeans(slika, centri, 10)
end = time.perf_counter()
print(f"Time taken: {end - start:0.6f} seconds")

start = time.perf_counter()
kmeans_original_colors(slika, centri, 10)
end = time.perf_counter()
print(f"Time taken: {end - start:0.6f} seconds")
'''

kmeans(slika, centri, 10)


cv2.waitKey(0)
cv2.destroyAllWindows()
