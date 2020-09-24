import cv2 as cv
import csv
import glob
import numpy as npy
from math import copysign, log10


def momentos_hu(nombre):
    # abro la imagen 'nombre'
    image = cv.imread(nombre)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # le aplico un threshold adaptativo y la invierto
    bin = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 67, 2)
    bin = 255 - bin

    # Le aplico las operaciones morphologicas para sacar ruido
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opening = cv.morphologyEx(bin, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    # Saco los contrnos y busco el mayor
    contornos, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contmax = max(contornos, key=cv.contourArea)

    # Calculo los momentos y momentos de hu y los paso a una ecala ams amnejable
    momentos = cv.moments(contmax)
    momentoshu = cv.HuMoments(momentos)
    for n in range(0, 6):
        momentoshu[n] = -1*copysign(1.0, momentoshu[n])*log10(abs(momentoshu[n]))

    # Devuelvo los momentos de hu de esa foto
    return momentoshu


def escribir_momentoshu(etiqueta, writer):
    # Guardo en fotos todos los archibos dentro de la carpeta de nombre etiqueta
    fotos = glob.glob('./shapes/' + etiqueta +'/*')

    # recorro todos las fotos, saco los momentos de hu con la funcion momentos_hu y los escribo en una fila con
    # el tipó de forma que es
    for nombre in fotos:
        fila = npy.append(momentos_hu(nombre), etiqueta)
        writer.writerow(fila)



def generar_mom_hu():
    # abro el archivo .csv donde escribo todos mis datos del dataset
    with open('generated-files/shapes-hu-moments.csv', 'w', newline='')as file:
        #genero mi escritor para escribir en dicho archivo
        writer = csv.writer(file)
        # llamo la funcion que escribe los datos y le paso el escritor y el nombre de la carpeta que debe analizar
        escribir_momentoshu('estrella', writer)
        escribir_momentoshu('triangulo', writer)
        escribir_momentoshu('rectangulo', writer)




