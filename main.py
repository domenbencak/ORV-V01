import cv2
import numpy as np
import scipy


def my_roberts_manual(slika):
    # Definicija Robertsovih jeder
    jedro_x = np.array([[1, 0], [0, -1]])
    jedro_y = np.array([[0, 1], [-1, 0]])

    # Pridobi velikost slike in jedra
    visina_slike, sirina_slike = slika.shape
    visina_jedra, sirina_jedra = jedro_x.shape

    # Inicializacija gradienta x in y na 0
    gradient_x = np.zeros_like(slika)
    gradient_y = np.zeros_like(slika)

    # Kovolucija jedra
    for y in range(visina_slike - visina_jedra + 1):
        for x in range(sirina_slike - sirina_jedra + 1):
            podslika = slika[y:y + visina_jedra, x:x + sirina_jedra]
            gradient_x[y + 1, x + 1] = np.sum(jedro_x * podslika)
            gradient_y[y + 1, x + 1] = np.sum(jedro_y * podslika)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_roberts(slika):
    # Definicija Robertsovih jeder
    jedro_x = np.array([[1, 0], [0, -1]])
    jedro_y = np.array([[0, 1], [-1, 0]])

    # Aplikacija jeder na sliko
    gradient_x = scipy.ndimage.convolve(slika, jedro_x)
    gradient_y = scipy.ndimage.convolve(slika, jedro_y)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_roberts_absolute(slika):
    # Definicija Robertsovih jeder
    jedro_x = np.array([[1, 0], [0, -1]])
    jedro_y = np.array([[0, 1], [-1, 0]])

    # Aplikacija jeder na sliko
    gradient_x = scipy.ndimage.convolve(slika, jedro_x)
    gradient_y = scipy.ndimage.convolve(slika, jedro_y)

    # Kalkulacija približne magnitude
    gradient_mag = np.absolute(gradient_x) + np.absolute(gradient_y)

    # Normalizacija gradienta magnitude
    gradient_mag *= np.uint8(255.0 / gradient_mag.max())
    gradient_mag = np.uint8(gradient_mag)

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_prewitt(slika):
    # Definicija Prewitt jeder
    jedro_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    jedro_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Aplikacija jeder na sliko
    gradient_x = scipy.ndimage.convolve(slika, jedro_x)
    gradient_y = scipy.ndimage.convolve(slika, jedro_y)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_prewitt_manual(slika):
    # Definicija Prewitt jeder
    jedro_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    jedro_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Pridobi velikost slike in jedra
    visina_slike, sirina_slike = slika.shape
    visina_jedra, sirina_jedra = jedro_x.shape

    # Inicializacija gradienta x in y na 0
    gradient_x = np.zeros_like(slika)
    gradient_y = np.zeros_like(slika)

    # Konvolucija jedra
    for y in range(visina_slike - visina_jedra + 1):
        for x in range(sirina_slike - sirina_jedra + 1):
            podslika = slika[y:y + visina_jedra, x:x + sirina_jedra]
            gradient_x[y + 1, x + 1] = np.sum(jedro_x * podslika)
            gradient_y[y + 1, x + 1] = np.sum(jedro_y * podslika)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_gaussian(slika, velikost_jedra=3, sigma=1.0):
    # Ustvari jedro podane velikost in z dano sigmo
    jedro = np.zeros((velikost_jedra, velikost_jedra))
    # Pridobi x in y koordinato sredine jedra (deljenje in zaokrožanje navzdol)
    center = velikost_jedra // 2
    print(f'center: {center}')
    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x, y = i - center, j - center
            jedro[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            print(f'jedro{i},{j}]: {jedro[i, j]}')
    jedro /= np.sum(jedro)

    # Obloži sliko z ničlami, da na robu ne pride do pokvaritve
    slika_padded = np.pad(slika, ((center, center), (center, center)), mode='constant')

    # Konvolucija jedra
    slika_gauss = np.zeros_like(slika)
    for y in range(slika.shape[0]):
        for x in range(slika.shape[1]):
            podslika = slika_padded[y:y + velikost_jedra, x:x + velikost_jedra]
            slika_gauss[y, x] = np.sum(podslika * jedro)

    return slika_gauss


def my_sobel(slika):
    # V bistvu kombinacija Prewitt in Gaussovega filtra za glajenje
    slika = my_gaussian(slika)
    # Definicija Sobel jeder
    jedro_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    jedro_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Aplikacija jeder na sliko
    gradient_x = scipy.ndimage.convolve(slika, jedro_x)
    gradient_y = scipy.ndimage.convolve(slika, jedro_y)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_sobel_manual(slika):
    # V bistvu kombinacija Prewitt in Gaussovega filtra za glajenje
    slika = my_gaussian(slika)
    # Definicija Sobel jeder
    jedro_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    jedro_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Pridobi velikost slike in jedra
    visina_slike, sirina_slike = slika.shape
    visina_jedra, sirina_jedra = jedro_x.shape

    # Inicializacija gradienta x in y na 0
    gradient_x = np.zeros_like(slika)
    gradient_y = np.zeros_like(slika)

    # Kovolucija jedra
    for y in range(visina_slike - visina_jedra + 1):
        for x in range(sirina_slike - sirina_jedra + 1):
            podslika = slika[y:y + visina_jedra, x:x + sirina_jedra]
            gradient_x[y + 1, x + 1] = np.sum(jedro_x * podslika)
            gradient_y[y + 1, x + 1] = np.sum(jedro_y * podslika)

    # Kalkulacija magnitude gradienta
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Normalizacija gradienta magnitude
    gradient_mag *= 255.0 / gradient_mag.max()

    # Pretvorba gradient magnitude v uint8
    gradient_mag = gradient_mag.astype(np.uint8)

    return gradient_mag


def my_canny(slika, sp_prag, zg_prag):
    spremenjena_slika = cv2.Canny(slika, threshold1=sp_prag, threshold2=zg_prag)
    return spremenjena_slika


def spremeni_kontrast_overflow(slika, alfa, beta):
    spremenjena_slika = alfa * slika + beta
    return spremenjena_slika


def spremeni_kontrast_manual(slika, alfa, beta):
    spremenjena_slika = slika * alfa
    for row in range(spremenjena_slika.shape[0]):
        for col in range(spremenjena_slika.shape[1]):
            pixel = spremenjena_slika[row, col]
            if pixel + beta > 255:
                spremenjena_slika[row, col] = 255
            elif pixel + beta < 0:
                spremenjena_slika[row, col] = 0
            else:
                spremenjena_slika[row, col] += beta

    return spremenjena_slika


def spremeni_kontrast_cv2(slika, alfa, beta):
    output = cv2.convertScaleAbs(slika, alpha=alfa, beta=beta)
    return output


def spremeni_kontrast(slika, alfa, beta):
    slika = slika.astype('float32')
    slika = alfa * slika + beta
    slika = np.clip(slika, 0, 255).astype('uint8')
    return slika


slika = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Slika', slika)
# slika2 = cv2.imread('lenna.png')
# cv2.imshow('Slika2', slika2)


adjusted_img = spremeni_kontrast(slika, alfa=1.5, beta=0)
cv2.imshow('Spremeni_kontrast 1.5, 0', adjusted_img)

# adjusted_img2 = my_gaussian(adjusted_img)
# cv2.imshow('Spremeni_kontrast 0.2, 0 + Gaussian blur[3]', adjusted_img2)


'''
adjusted_img2 = spremeni_kontrast_cv2(slika, alfa=2, beta=50)
cv2.imshow('Adjusted2', adjusted_img2)

adjusted_img3 = spremeni_kontrast(slika, alfa=2, beta=50)
cv2.imshow('Adjusted3', adjusted_img3)


kontrast_slika1 = spremeni_kontrast(slika, alfa=1.5, beta=0)
# cv2.imshow('Konstrast slika', kontrast_slika)

kontrast_slika9 = spremeni_kontrast(slika, alfa=3, beta=0)
# cv2.imshow('Konstrast slika2', kontrast_slika2)



prewitt = my_prewitt_manual(slika)
cv2.imshow('Prewitt Original', prewitt)
prewitt2 = my_prewitt_manual(adjusted_img)
cv2.imshow('Prewitt Manual', prewitt2)
'''
'''
sobel = my_sobel(slika)
cv2.imshow('Sobel Original', sobel)
sobel2 = my_sobel_manual(adjusted_img)
cv2.imshow('Sobel Manual', sobel2)
'''
'''
roberts = my_roberts_manual(slika)
cv2.imshow('Roberts Original', roberts)
roberts2 = my_roberts_manual(adjusted_img)
cv2.imshow('Roberts 1, 120', roberts2)
'''
'''
lowerb1 = 10
higherb1 = 30

lowerb2 = 10
higherb2 = 30

canny = my_canny(slika, lowerb1, higherb1)
cv2.imshow(f'Canny {lowerb1} {higherb1}', canny)

canny2 = my_canny(adjusted_img, lowerb2, higherb2)
cv2.imshow(f'Canny1 {lowerb2} {higherb2}', canny2)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
