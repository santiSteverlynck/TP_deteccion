import cv2 as cv
import numpy as npy
import csv
from Tp_deteccion.covertidoretiquetas import etiqueta_num


def cargar_datosentre():
    # genero los vectores para los datos de entrenamiento y las atiquetas de als formas
    dataentre = []
    etientre = []

    # Abro el archivo que tiene todos los datos del dataset
    with open('generated-files/shapes-hu-moments.csv')as csv_file:

        # leo todos los datos del archivo
        csv_lector = csv.reader(csv_file, delimiter=',')
        for fila in csv_lector:
            etiqueta = fila.pop()  # guardo el ultimo valor de la linea pq es la etiqueta
            # Guardo los datos de los momento sen la variable floats y
            # convierto las etiquetas a numeros y la guardo tambien
            floats = []
            for n in fila:
                floats.append(float(n))
            dataentre.append(npy.array(floats, dtype=npy.float32))
            etientre.append(npy.array(etiqueta_num(etiqueta), dtype=npy.int32))

    dataentre = npy.array(dataentre, dtype=npy.float32)
    etientre = npy.array(etientre, dtype=npy.int32)
    # Devuelvo los arrays con los datos para entrenamiento
    return dataentre, etientre


def entrenamientodt():  # entrenop un arbol de decision
    # llamo la funcion apra cargar los datos de entrenamiento
    dataentre, etientre = cargar_datosentre()

    # genero el arbol con dichos datos
    tree = cv.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(dataentre, cv.ml.ROW_SAMPLE, etientre)
    return tree


def entrenamientobayes():  # otros tipo de maquina
    dataentre, etientre = cargar_datosentre()
    bayes = cv.ml.NormalBayesClassifier_create()
    bayes.train(dataentre, cv.ml.ROW_SAMPLE, etientre)
    return bayes


def entrenamientosvm():
    dataentre, etientre = cargar_datosentre()
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.trainAuto(dataentre, cv.ml.ROW_SAMPLE, etientre)
    return svm
