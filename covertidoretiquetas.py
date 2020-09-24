
# Paso las etiquetas a numeros para el entrenamiento
def etiqueta_num(etiqueta):
    if etiqueta == 'estrella':
        return 1
    if etiqueta == 'rectangulo':
        return 2
    if etiqueta == 'triangulo':
        return 3
    else:
        raise Exception('No identificado')


# Mi modelo devuelve numeros entonces los vuelvo a sus etiquetas
def num_etiqueta(num):
    if num == 1:
        return 'estrella'
    if num == 2:
        return 'rectangulo'
    if num == 3:
        return 'triangulo'
    else:
        raise Exception('No idetificado')
