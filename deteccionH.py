import cv2 as cv

from Tp_deteccion.captura import captura, filtrar
from math import copysign, log10


def on_trackbar_change(val):
    pass


window_name = 'Detector'
cv.namedWindow(window_name)

# genero los trackbars para threshold, differencia y areas
cv.createTrackbar('Tresh', window_name, 70, 255, on_trackbar_change)
cv.createTrackbar('Diff', window_name, 0, 150, on_trackbar_change)
cv.createTrackbar('Area MAX', window_name, 50, 100, on_trackbar_change)
cv.createTrackbar('Area MIN', window_name, 0, 25, on_trackbar_change)


def deteccion():
    # capturo el video del telefono(1) o de la computadora(0)
    cap = cv.VideoCapture(1)
    capt = ()  # inicializo la variable para poder usarla
    momhu_positivo = []

    while True:
        #  leo la imagen de la camara y la comvierto a gris
        #  despues de vuelta a RGB (display) para poder pintar en color
        _, image = cap.read()
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        display = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

        # obtengo los valores de los trackbars
        threshold = cv.getTrackbarPos('Tresh', window_name)
        differencia = cv.getTrackbarPos('Diff', window_name)
        area_max = cv.getTrackbarPos('Area MAX', window_name)
        area_min = cv.getTrackbarPos('Area MIN', window_name)

        # aplico el threshold a la imagen
        _, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

        # aplico las operaciones morfologicas
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # Busco los contornos
        contornos, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # flitro los contornos por tamaño de area
        contornos_filtrados = filtrar(contornos, area_max, area_min)

        # Cuando toco la tecla 'c' capturo el contorno mas grande y printeo sus momentos de hu
        if cv.waitKey(1) & 0xFF == ord('c'):
            capt = captura(contornos_filtrados)
            print('Momentos del contorno capturado')
            momentos = cv.moments(capt)
            mhc = cv.HuMoments(momentos)

            for n in range(0, 6):
                mhc[n] = -1*copysign(1.0, mhc[n])*log10(abs(mhc[n]))
                print(str(mhc[n]))

        # Recorro los contornos filtrados y los comapro con el capturado
        for cont in contornos_filtrados:
            if cv.matchShapes(capt, cont, cv.CONTOURS_MATCH_I2, 0) < (differencia/100):
                # si el contorno es parecido lo dibujo en verde y guardo sus momentos de HU
                cv.drawContours(display, cont, -1, (0, 255, 0), 3)
                mom_positivo = cv.moments(cont)
                momhu_positivo = cv.HuMoments(mom_positivo)

            else:
                cv.drawContours(display, cont, -1, (0, 0, 255), 2)  # si el contorno es distinto los dibujo en rojo

        cv.imshow(window_name,  display)

        # Si aprieto la tecla 'H' muestro los momentos del contorno que es parecido
        if cv.waitKey(1) & 0xFF == ord('h'):
            print('Momentos del contorno comparado')
            for n in range(0, 6):
                momhu_positivo[n] = -1 * copysign(1.0, momhu_positivo[n]) * log10(abs(momhu_positivo[n]))
                print(str(momhu_positivo[n]))

        # si aprieto la tecla 'q' termino el programa
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


deteccion()
