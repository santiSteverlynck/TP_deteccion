import cv2 as cv

# capturar el mayor contorno

def captura (contornos):

    captura = contornos[0]  # Elijo el primer contorno para comparar
    for contorno in contornos:
        if cv.contourArea(contorno) > cv.contourArea(captura):
            captura = contorno  # si el c0ontorno es mayor al capturado lo sobre escribo

    return captura


# filrar los contornos segun el area
def filtrar(contornos, MAX, MIN):

    cont = []   # inicializo la variable
    # Recorro todos los contronos existentes  y me fijo si estan en el rango
    for n in contornos:
        if float(MIN*1000) < float(cv.contourArea(n)) < float(MAX*1000):
            cont.append(n)  #Guardo el contorno en cont

    return cont