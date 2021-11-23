#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:45:17 2021

@author: angel
"""

import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import time

np.random.seed(1)

print("Ejercicio 1A-C")


def pintarresultados (imagenes,titulos):
    fig = plt.figure(figsize=(10,5))
    ax = []
    
    #plt.close(1)
    
    cols = len(imagenes)
    filas = 1

    for i in range(1, cols*filas+1):
        ax.append(fig.add_subplot(filas,cols,i))
        
        ax[-1].set_title(titulos[i-1])
        
        if imagenes[i-1].ndim == 3 and imagenes[i-1].shape[2] >= 3:
            plt.imshow(imagenes[i-1][:,:,::-1])
        else:
            plt.imshow(imagenes[i-1], cmap='gray')
            
    plt.show()
    
def mostrarimagen(imagen,titulo=""):
    plt.figure(figsize=(10,5))
    plt.title(titulo)
    #Comprobamos si la imagen es tribanda
    if imagen.ndim == 3 and imagen.shape[2] >= 3:
        # Si es tribanda tenemos que recorrer la representación
        # BGR al revés para que sea RGB
        plt.imshow(imagen[:,:,::-1])
        
    else:
        # Si es monobonda se lo indicamos a matplotlib
        plt.imshow(imagen,cmap='gray')
        
    plt.show()
        
"""
Funcion gaussiana para un punto x y un sigma concreto
"""
def gaussiana (x,sigma):
    return math.exp( - (x**2) / ( 2*sigma**2 ))


"""
Primera derivada de la gaussiana para un punto x y un sigma concreto
"""
def primeraGaussiana(x,sigma):
    return ( - (x * ( math.exp( - (x**2) / (2*sigma**2))) ) / (sigma**2) )


"""
Segunda derivada de la gaussiana parsigmaa un punto x y un sigma concreto
"""
def segundaGaussiana(x,sigma):
    return ( ( (x**2 - sigma**2) * math.exp( - ( x**2 / (2*sigma**2) )) ) / sigma**4 )


def leeimagen(filename, flagColor):
    img = cv.imread(filename,flagColor).astype('float64')
    
    return img
    
    
def normalizarmatriz(imagen):
    # Convertimos la imagen a tribanda si esta no lo es
    img_normalizada = imagen.astype(float)
    
    canales = 1
    
    if (img_normalizada.ndim == 3):
        canales = 3
        
    # Normalizamos cada canal de la imagen
    for canal in range(canales):
        # Obtenemos el minimo y el maximo de cada canal
        if(canales == 3):
            minimo = np.min(img_normalizada.transpose(2,0,1)[canal])
            maximo = np.max(img_normalizada.transpose(2,0,1)[canal])
            
            # Comprobamos si dividimos por 0, si no, normalizamos restando el minimo y diviendiendo entre la
            # diferencia entre maximo y minimo y si sí simplemente restamos el minimo sin dividir
            if maximo - minimo != 0:
                img_normalizada.transpose(2,0,1)[canal] = (img_normalizada.transpose(2,0,1)[canal] - minimo) / (maximo - minimo)
            else:
                img_normalizada.transpose(2,0,1)[canal] = img_normalizada.transpose(2,0,1)[canal] - minimo
        else:
            minimo = np.min(img_normalizada)
            maximo = np.max(img_normalizada)
            
            # Comprobamos si dividimos por 0, si no, normalizamos restando el minimo y diviendiendo entre la
            # diferencia entre maximo y minimo y si sí simplemente restamos el minimo sin dividir
            if maximo - minimo != 0:
                img_normalizada = (img_normalizada - minimo) / (maximo - minimo)
            else:
                img_normalizada = img_normalizada - minimo
    return img_normalizada
"""
Funcion que calcula un kernel gaussiano 1D ado un sigma o un tamaño de kernel
"""
def kernelGaussiano1D (func=gaussiana,sigma=None,tam=None):
    
    # Comprobamos el dato que nos dan y calculamos
    # el que falte. Si no nos dan ninguno avisamos
    # al usuario de que tiene que pasar al menos 1
    # y paramos el programa
    if (sigma != None):
        tam = 2 * 3 * sigma + 1
    elif (tam != None):
        sigma = (tam - 1) / 6
    else:
        assert("Debe pasar un valor de sigma o un tamaño de mascara")
        
    kernel = []
    
    mitad_intervalo = int( np.floor(tam/2) )
    
    # Añadimos los valores del kernel
    for i in range (-mitad_intervalo, mitad_intervalo + 1):
        kernel.append( func(i,sigma) )
    
    # Si el kernel es gaussiano (no es ninguna derivada de este)
    # lo normalizamos
    
    kernel_normalizado = np.array(kernel)
    
    if ( func == gaussiana ):
        kernel_normalizado = kernel_normalizado / (np.sum(kernel))
    elif ( func == primeraGaussiana):
        kernel_normalizado = sigma * kernel_normalizado
    elif ( func == segundaGaussiana):
        kernel_normalizado = (sigma**2) * kernel_normalizado
        
    
    return kernel_normalizado

def aniade_bordes(imagen,mascara,tipo_borde):    
    borde = int( (len(mascara) -1) / 2)     
    imagen_borde = cv.copyMakeBorder(imagen, borde, borde, borde, borde, tipo_borde)

    return imagen_borde


def convulcionar (imagen,kernel_horizontal,kernel_vertical=None):    
    if (kernel_vertical is None):
        kernel_vertical = kernel_horizontal
        
    tam_borde = len(kernel_horizontal) // 2
    
    mascara_horizontal = (np.tile(kernel_horizontal,(imagen.shape[1],1))).T
    mascara_vertical = (np.tile(kernel_vertical,(imagen.shape[1],1))).T
    
    canales = 1
    
    if (imagen.ndim == 3):
        canales = 3
    
    imagen_tmp = []
    cont = 0
    for canal in range(canales):
        
        if (canales == 3):
            imagen_tmp.append([])
            
        for i in range(tam_borde, (imagen.shape[0]-tam_borde)):
            if (canales == 3):
                tmp = np.multiply(imagen[i-tam_borde:i+tam_borde+1:,:,canal ],mascara_horizontal)
                tmp = np.sum(tmp,axis=0)
                imagen_tmp[cont].append(tmp)
            else:
                tmp = np.multiply(imagen[i-tam_borde:i+tam_borde+1: ],mascara_horizontal)
                tmp = np.sum(tmp,axis=0)
                imagen_tmp.append(tmp)
                
        cont+=1
    
    imagen_resultado = []
    imagen_tmp = np.array(imagen_tmp)
    imagen_tmp = imagen_tmp.T
    
    mascara_horizontal = (np.tile(kernel_horizontal,(imagen_tmp.shape[1],1))).T
    mascara_vertical = (np.tile(kernel_vertical,(imagen_tmp.shape[1],1))).T
    cont = 0
    for canal in range(canales):
        if (canales == 3):
            imagen_resultado.append([])
            
        for i in range(tam_borde,(imagen_tmp.shape[0] - tam_borde)):
            if (canales == 3):
                tmp = np.multiply(imagen_tmp[i-tam_borde:i+tam_borde+1:,:,canal ],mascara_vertical)
                tmp = np.sum(tmp,axis=0)
                imagen_resultado[cont].append(tmp)
            else:
                tmp = np.multiply(imagen_tmp[i-tam_borde:i+tam_borde+1: ],mascara_vertical)
                tmp = np.sum(tmp,axis=0)
                imagen_resultado.append(tmp)
                
        cont+=1
            
    imagen_resultado = np.array(imagen_resultado)

    return imagen_resultado.T

def calcular_escalas(imagen,num_escalas):
    
    octava = []
    octava.append(imagen)
    
    for i in range(1,num_escalas+1):
        sigma = 1.6 * math.sqrt( 2**((2*i)/3) - 2**((2*(i-1))/3))
        imagen_convol = octava[-1]
        mascara_convol = kernelGaussiano1D(sigma=sigma)
        imagen_convol = aniade_bordes(imagen_convol, mascara_convol, cv.BORDER_REFLECT)
        next_escala = convulcionar(imagen_convol, mascara_convol)
        octava.append(next_escala)
        
    return np.array(octava)

def calcular_octavas(imagen, num_octavas):
    piramide = []
    imagen = cv.resize(imagen,dsize=(imagen.shape[1]*2,imagen.shape[0]*2),interpolation=cv.INTER_LINEAR)
    
    for i in range(0,num_octavas):
        nueva_octava = calcular_escalas(imagen,5)
        piramide.append(nueva_octava)
        imagen = imagen[::2,::2]
        
    
    return piramide


def obtener_escalas_DoG (lista_escalas):
    lista_DoG = []
    
    for i in range(1, len(lista_escalas)):
        lista_DoG.append(lista_escalas[i] - lista_escalas[i-1])
    
    return np.array(lista_DoG)

def octavas_DoG (lista_octavas):
    lista_octavas_DoG = []
    
    for i in range(0, len(lista_octavas)):
        lista_octavas_DoG.append(obtener_escalas_DoG(lista_octavas[i]))
    
    return (lista_octavas_DoG)


def obtener_octavas_DoG (imagen_ini, num_octavas_totales, num_escalas_totales, sigma_original):
    lista_octavas = calcular_octavas(imagen_ini, num_octavas_totales)
    
    return octavas_DoG(lista_octavas)



def get_sigmak(sigma_original, num_escalas_totales, octava, escala):
    k = -num_escalas_totales + num_escalas_totales*octava + escala
    
    return sigma_original* 2**(k / num_escalas_totales)

def mostrar_im_en_bloque(vim, titulos, nrows=1, normalizar=False, tam_fig=(10, 5), titulo_fig=None): 
    fig = plt.figure(figsize=tam_fig)
    
    if nrows == 1:
        ncols = len(vim)
    else:
        ncols = math.ceil(len(vim) / nrows)
    
    for i in range(len(vim)):
        imagen = vim[i]
        
        fig.add_subplot(nrows, ncols, i+1)
        
        plt.axis('off')
        if imagen.ndim == 3:
            if normalizar:
                plt.axis('off')
                plt.imshow(normalizarmatriz(imagen)[:,:,::-1])
            else:
                plt.axis('off')
                plt.imshow(imagen[:,:,::-1])
            
            plt.title(titulos[i])
        else:
            plt.imshow(imagen, cmap='gray')
            plt.title(titulos[i])
    
    if not titulo_fig is None:
        plt.axis('off')
        fig.suptitle(titulo_fig())
    
    fig.tight_layout()   
    plt.show()

def mostrar_escalas_gaussianas (lista_octavas, num_escalas_totales):
    lista_titulos = []
    for i in range(0, len(lista_octavas)):
        for j in range(1, num_escalas_totales+1):
            lista_titulos.append("Octava " + str(i) + " Escala " + str(j))
    
    flat_list_octavas = [item for sublist in lista_octavas for item in sublist[1:(num_escalas_totales+1)]]
    
    mostrar_im_en_bloque(flat_list_octavas, lista_titulos, nrows=len(lista_octavas)+1, tam_fig=(6, 12))
    
def mostrar_escalas_DoG (lista_octavas):
    lista_titulos = []
    for i in range(0, len(lista_octavas)):
        for j in range(1, len(lista_octavas[i])+1):
            lista_titulos.append("Octava " + str(i) + " Escala " + str(j))
    
    flat_list_octavas = [item for sublist in lista_octavas for item in sublist]
    
    mostrar_im_en_bloque(flat_list_octavas, lista_titulos, nrows=len(lista_octavas), tam_fig=(8, 8))
    
    
def extraer_keypoints (imagen_ini, num_octavas_totales, num_escalas_totales,
                       sigma_original, num_keypoints=100, radio=6):
    
    diameter = radio * 2
    
    pir_lap = obtener_octavas_DoG(imagen_ini, num_octavas_totales, num_escalas_totales, sigma_original)
    
    # Vector de posibles keypoints: se guarda la posicion del pixel, su diametro,
    # su respuesta y la octava donde se identifico
    aux = []
    dtype_kp = [('x', float), ('y', float), ('size', float), ('response', float),
                           ('abs_response', float), ('octave', float)]
    kp = np.array(aux, dtype=dtype_kp)
    
    
    for num_oct in range(0, len(pir_lap)):
        octava = pir_lap[num_oct]
        mascara_opt = np.zeros(octava[0].shape)
        marco_fil = octava[0].shape[0] - 1
        marco_col = octava[0].shape[1] - 1
        
        for num_esc in range(1, len(pir_lap[0])-1):
            for j in range(1, marco_col):
                for i in range(1, marco_fil):
                    if mascara_opt[i][j] == 0:
                        index_max_local = np.argmax(octava[num_esc-1:num_esc+2, i-1:i+2, j-1:j+2])
                        
                        if index_max_local == 13:
                            sigmak = get_sigmak(sigma_original, num_escalas_totales, num_oct, num_esc)
                            kp = np.append( kp,np.array((i*2**(num_oct-1), j*2**(num_oct-1),
                                            diameter*sigmak,octava[num_esc][i][j],abs(octava[num_esc][i][j]),
                                            num_oct),dtype=dtype_kp) )
                            
                            mascara_opt[i-1:i+2,j-1:j+2] = 1
    
    return kp


def aniadir_keypoints_im (imagen_ini, array_keypoints): 
    im_keypoints = cv.drawKeypoints(imagen_ini.astype('uint8'), array_keypoints, np.array([]),
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_keypoints


##############################################################################
imagen = leeimagen('imagenes/Yosemite1.jpg', 0)
imagen2 = leeimagen('imagenes/Yosemite2.jpg',0)

pir_lowe = calcular_octavas(imagen, 4)

pir_lowe2 = calcular_octavas(imagen2,4)

mostrar_escalas_gaussianas(pir_lowe, 3)
mostrar_escalas_gaussianas(pir_lowe2,3)

print('Ejercicio 1 D-E.')

imagen = leeimagen('imagenes/Yosemite1.jpg', 0)
imagen2 = leeimagen('imagenes/Yosemite2.jpg',0)

num_escalas_totales = 3
num_octavas_totales = 4
sigma_original = 1.6

num_keypoints = 100
radio = 6

array_keypoints1 = extraer_keypoints (imagen, num_octavas_totales, num_escalas_totales,
                                       sigma_original, num_keypoints, radio)


# Ordeno el vector por respuesta
array_keypoints1 = np.sort(array_keypoints1, order='abs_response')

# Guardar los 100 keypoints con mas respuesta
if len(array_keypoints1) > num_keypoints:
    array_keypoints1 = array_keypoints1[-num_keypoints:]

lista_keypoints = [cv.KeyPoint(x=float(math.ceil(kp['y'])), y=float(math.ceil(kp['x'])),
                               size=kp['size'], response=kp['response'],
                               octave=int(kp['octave']))
                   for kp in array_keypoints1]

array_keypoints=np.array(lista_keypoints)

yosemite1_keypoints = aniadir_keypoints_im(imagen, array_keypoints)


mostrarimagen(yosemite1_keypoints, "Yosemite 1 con keypoints")

input("\n--- Pulsar tecla para continuar ---\n")

array_keypoints2 = extraer_keypoints (imagen2, num_octavas_totales, num_escalas_totales,
                                      sigma_original, num_keypoints, radio)


# Ordeno el vector por respuesta
array_keypoints2 = np.sort(array_keypoints2, order='abs_response')

# Guardar los 100 keypoints con mas respuesta
if len(array_keypoints2) > num_keypoints:
    array_keypoints2 = array_keypoints2[-num_keypoints:]

# Construir vector de keypoints
# Como al mostrar la imagen se muestra invertida entonces las coordenadas de los puntos han de ser invertidas
lista_keypoints = [cv.KeyPoint(x=float(math.ceil(kp['y'])), y=float(math.ceil(kp['x'])),
                               size=kp['size'], response=kp['response'],
                               octave=int(kp['octave']))
                   for kp in array_keypoints2]

array_keypoints2 = np.array(lista_keypoints)


yosemite2_keypoints = aniadir_keypoints_im(imagen2, array_keypoints2)

mostrarimagen(yosemite2_keypoints, "Yosemite 2 con keypoints")

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 2')

def FuerzaBruta(imagen1,imagen2,num_matches):
    sift = cv.SIFT_create()
    
    kpts1, desc1 = sift.detectAndCompute(imagen1.astype('uint8'), None)
    kpts2, desc2 = sift.detectAndCompute(imagen2.astype('uint8'), None)
    
    bf = cv.BFMatcher( crossCheck=True)

    matches = bf.match(desc1,desc2)
    
    im_final = cv.drawMatches(imagen1.astype('uint8'), kpts1,
                               imagen2.astype('uint8'), kpts2,
                               random.sample(matches, num_matches),
                               np.array([]), flags=2)
    
    return im_final

def coincidenciasLowe2NN (imagen1,imagen2,num_matches=100):
    sift = cv.SIFT_create()
    
    kpts1, desc1 = sift.detectAndCompute(imagen1.astype('uint8'), None)
    kpts2, desc2 = sift.detectAndCompute(imagen2.astype('uint8'), None)
    
    bf = cv.BFMatcher()
    coincidencias = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    coincidencias_reales = []
    for coincidencias_x,coincidencias_y in coincidencias:
        if coincidencias_x.distance < 0.8*coincidencias_y.distance:
            coincidencias_reales.append(coincidencias_x)
            
    return kpts1, kpts2, coincidencias_reales

def drawMatchesLowe2NN(imagen1, imagen2, num_matches):
    kpts1, kpts2, matches = coincidenciasLowe2NN(imagen1, imagen2)
        
    im_final = cv.drawMatches(imagen1.astype('uint8'), kpts1,
                              imagen2.astype('uint8'), kpts2,
                              random.sample(matches, num_matches), np.array([]), flags=2)
    
    return im_final


imagen = leeimagen('imagenes/Yosemite1.jpg', 0)
imagen2 = leeimagen('imagenes/Yosemite2.jpg',0)

img_fuerza_bruta = FuerzaBruta(imagen, imagen2, 100)
img_Lowe2NN = drawMatchesLowe2NN(imagen,imagen2,100)

mostrarimagen(img_fuerza_bruta,'Correspondencias Yosemite con Fuerza Bruta')
mostrarimagen(img_Lowe2NN,'Correspondencias Yosemite con Lowe2NN')

input("\n--- Pulsar tecla para continuar ---\n")

print("Ejercicio 3")

def get_canvas(lista_im):
    len_y = 0
    len_x = 0

    for i in range(0, len(lista_im)):
        len_y = len_y + lista_im[i].shape[1]
        len_x = len_x + lista_im[i].shape[0]
            
    return np.zeros((len_x, len_y,3))

def recortar_canvas(canvas):
    diferentes_cero = np.nonzero(canvas)
    
    primera_columna = np.min(diferentes_cero[0])
    primera_fila = np.min(diferentes_cero[1])
    
    ultima_columna  = np.max(diferentes_cero[0]) + 1
    ultima_fila = np.max(diferentes_cero[1]) + 1
    
    canvas_final = np.delete(canvas, np.arange(ultima_columna, canvas.shape[0]), 0)
    canvas_final = np.delete(canvas_final, np.arange(ultima_fila, canvas.shape[1]), 1)
    
    canvas_final = np.delete(canvas_final, np.arange(0, primera_columna), 0)
    canvas_final = np.delete(canvas_final, np.arange(0, primera_fila), 1)
    
    return canvas_final

def construirmosaico(imagenes):
    # Indice del medio de la lista
    index_centro = math.ceil(len(imagenes) / 2) - 1
    
    # Dimensiones de la imagen central
    x_im_centro = imagenes[index_centro].shape[0]
    y_im_centro = imagenes[index_centro].shape[1]
    
    # Obtener el canvas inicial y sus dimensiones
    canvas_fondo = get_canvas(imagenes)
    
    x_canvas = canvas_fondo.shape[0]
    y_canvas = canvas_fondo.shape[1]
    
    # Homografia que transforma la imagen del medio al canvas
    h0 = np.array([[1, 0, y_canvas//2 - y_im_centro//2],
                         [0, 1, x_canvas//2 - x_im_centro//2],
                         [0, 0, 1]],
                        dtype=np.float64)
    
    ind_izq = 0
    ind_dcha = len(imagenes) - 1
    lista_h_izq = []
    lista_h_dcha = []

    # Crear homografia del extremo izquierdo hasta el centro
    while(ind_izq < index_centro):
        kpts1, kpts2, matches = coincidenciasLowe2NN(imagenes[ind_izq], imagenes[ind_izq+1])
        
        im_siguiente_pts = np.float32([ kpts1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        im_centro_pts = np.float32([ kpts2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        h_relativa, mask = cv.findHomography(im_siguiente_pts, im_centro_pts, cv.RANSAC)
        
        lista_h_izq.append(h_relativa)
        
        ind_izq+=1
        
    
    # Crear homografias del extremo derecho hasta el centro
    while(ind_dcha > index_centro):
        kpts_d1, kpts_d2, matches = coincidenciasLowe2NN(imagenes[ind_dcha], imagenes[ind_dcha-1])
        im_siguiente_pts = np.float32([ kpts_d1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        im_centro_pts = np.float32([ kpts_d2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        h_relativa, mask = cv.findHomography(im_siguiente_pts, im_centro_pts, cv.RANSAC)
        
        lista_h_dcha.append(h_relativa)
        
        ind_dcha-=1
    
    lista_h_izq.append(h0)
    lista_h_dcha.append(h0)
    
    # Componer todas las imagenes desde la izquierda
    ind_izq = 0
    
    
    # Si ES un mosaico de 3 imagenes
    if (len(lista_h_izq) < 2):
        h_total_izq = np.dot(h0,lista_h_izq[ind_izq])
        canvas_fondo = cv.warpPerspective(imagenes[ind_izq], h_total_izq, (y_canvas,x_canvas), dst=canvas_fondo, borderMode=cv.BORDER_TRANSPARENT)
        
          
    while(ind_izq < len(lista_h_izq) -1):
        i_aux = ind_izq +1
    
        h_total_izq = np.dot(lista_h_izq[i_aux],lista_h_izq[ind_izq])
        i_aux+=1
        
        while(i_aux < len(lista_h_izq)):
            h_total_izq = np.dot(lista_h_izq[i_aux],h_total_izq)
            i_aux+=1
        
        canvas_fondo = cv.warpPerspective(imagenes[ind_izq], h_total_izq, (y_canvas,x_canvas), dst=canvas_fondo, borderMode=cv.BORDER_TRANSPARENT)
        ind_izq+=1
    
    ind_dcha = 0
    ind_dcha_imagenes = len(imagenes) -1
    # Si ES un mosaico de 3 imagenes
    if (len(lista_h_dcha) < 2):
        h_total_dcha = np.dot(h0,lista_h_dcha[ind_dcha])
        canvas_fondo = cv.warpPerspective(imagenes[ind_dcha_imagenes], h_total_dcha, (y_canvas,x_canvas), dst=canvas_fondo, borderMode=cv.BORDER_TRANSPARENT)
         
    while (ind_dcha < len(lista_h_dcha)-1):
        i_aux = ind_dcha + 1
        
        h_total_dcha = np.dot(lista_h_dcha[i_aux],lista_h_dcha[ind_dcha])
        i_aux+=1
        
        while(i_aux < len(lista_h_dcha)):
            h_total_dcha = np.dot(lista_h_dcha[i_aux],h_total_dcha)
            i_aux+=1
        
        canvas_fondo = cv.warpPerspective(imagenes[ind_dcha_imagenes], h_total_dcha, (y_canvas,x_canvas), dst=canvas_fondo, borderMode=cv.BORDER_TRANSPARENT)
        ind_dcha+=1
        ind_dcha_imagenes-=1
        
    canvas_fondo = cv.warpPerspective(imagenes[index_centro], h0,(y_canvas,x_canvas), dst=canvas_fondo, borderMode=cv.BORDER_TRANSPARENT)
    
    return recortar_canvas(canvas_fondo)
    

imagen1 = leeimagen('imagenes/IMG_20211030_110413_S.jpg', 1)
imagen2 = leeimagen('imagenes/IMG_20211030_110415_S.jpg', 1)
imagen3 = leeimagen('imagenes/IMG_20211030_110417_S.jpg', 1)

imagenes = [imagen1,imagen2,imagen3]

mosaico_final = construirmosaico(imagenes)

mosaico_final = normalizarmatriz(mosaico_final)

mostrarimagen(mosaico_final,'Mosaico 3 imagenes')

input("\n--- Pulsar tecla para continuar ---\n")

print("Bonus 2 A")
imagen1 = leeimagen('imagenes/IMG_20211030_110410_S.jpg', 1)
imagen2 = leeimagen('imagenes/IMG_20211030_110413_S.jpg', 1)
imagen3 = leeimagen('imagenes/IMG_20211030_110415_S.jpg', 1)
imagen4 = leeimagen('imagenes/IMG_20211030_110417_S.jpg', 1)
imagen5 = leeimagen('imagenes/IMG_20211030_110418_S.jpg', 1)
imagen6 = leeimagen('imagenes/IMG_20211030_110420_S.jpg', 1)
imagen7 = leeimagen('imagenes/IMG_20211030_110421_S.jpg', 1)
imagen8 = leeimagen('imagenes/IMG_20211030_110425_S.jpg', 1)
imagen9 = leeimagen('imagenes/IMG_20211030_110426_S.jpg', 1)
imagen10 = leeimagen('imagenes/IMG_20211030_110428_S.jpg', 1)
imagen11 = leeimagen('imagenes/IMG_20211030_110431_S.jpg', 1)
imagen12 = leeimagen('imagenes/IMG_20211030_110433_S.jpg', 1)
imagen13 = leeimagen('imagenes/IMG_20211030_110434_S.jpg', 1)
imagen14 = leeimagen('imagenes/IMG_20211030_110436_S.jpg', 1)

lista_mosaico = [imagen1,imagen2,imagen3,imagen4,imagen5,imagen6,imagen7,imagen8,imagen9,imagen10,imagen11,imagen12,imagen13,imagen14]


mosaico_final = construirmosaico(lista_mosaico)

mosaico_final = normalizarmatriz(mosaico_final)

mostrarimagen(mosaico_final, "Mosaico")