import cv2 as cv
import numpy as npy

from Tp_deteccion.MomHU import generar_mom_hu
from Tp_deteccion.entrenamiento import entrenamientodt, entrenamientosvm, entrenamientobayes
from Tp_deteccion.covertidoretiquetas import num_etiqueta
from math import copysign, log10


def detector():
    cap = cv.VideoCapture(1)
    generar_mom_hu()  # llamo la funcion que escribe los datos para el entrenamiento
    modelo = entrenamientodt()  # guardo en miodelo mi arbol de decision
    # modelo = entrenamientosvm  # gaurdo en modelo el svm
    # modelo = entrenamientobayes  # guardo en modelo el bayes
    while True:
        _, image = cap.read()
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Aplico un threshold adaptativo
        bin = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 67, 2)
        # invierto el thresh para que funcione emjor con el findContours
        bin = 255 - bin

        # Aplico las opcione morphologicas
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        opening = cv.morphologyEx(bin, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # Hallo los contornos de mi video
        contornos, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contmax = max(contornos, key=cv.contourArea)  # busco el contorno ams grande

        # saco los momentos y momentos de hu y los paso a una escala mas trabajable
        momentos = cv.moments(contmax)
        momentoshu = cv.HuMoments(momentos)
        for n in range(0, 6):
            momentoshu[n] = -1 * copysign(1.0, momentoshu[n]) * log10(abs(momentoshu[n]))

        # creo miu muestra y pido la prediccion al modelo
        muestra = npy.array([momentoshu], dtype=npy.float32)
        respuesta = modelo.predict(muestra)[1]
        # dibujo el contorno mas gradne y su prediccion
        cv.drawContours(image, contmax, -1, (255, 0, 0), 3)
        cv.putText(image, num_etiqueta(respuesta), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('detector', image)
        # cv.imshow('hla',closing)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


detector()
