#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:23:40 2021

@author: angel
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Ejercicio 1.- Escribir una función que lea el fichero de una imagen y
# permita mostrarla tanto en grises como en color
def leeimagen(filename, flagColor):
    img = cv.imread(filename,flagColor)
    
    return img
    

def mostrarimagen(imagen):
    #Comprobamos si la imagen es tribanda
    if imagen.ndim == 3 and imagen.shape[2] >= 3:
        # Si es tribanda tenemos que recorrer la representación
        # BGR al revés para que sea RGB
        plt.imshow(imagen[:,:,::-1])
        plt.show()
    else:
        # Si es monobonda se lo indicamos a matplotlib
        plt.imshow(imagen,cmap='gray')
        plt.show()

print ("Ejercicio 1")

# Pintamos la imagen en escala de grises
flagColor = 0
imagen1 = leeimagen("./images/orapple.jpg",flagColor)
mostrarimagen(imagen1)

# Pintamos la imagen en color
flagColor = 1
imagen = leeimagen("./images/orapple.jpg",flagColor)
mostrarimagen(imagen)

input("\n-------Pulsa una tecla para continuar-------\n")

print("Ejercicio 2")
# Ejercicio 2.- Escribir una función que permita visualizar una
# matriz de números reales arbitraria tanto monobanda como tribanda

def normalizarmatriz(matriz):
    matriz_normalizada = matriz.copy()
    
    # Sacamos el máximo y el mínimo de la matriz
    maximo = np.max(matriz_normalizada)
    minimo = np.min(matriz_normalizada)
    
    # La normalizamos restando el mínimo a toda la matriz y diviendiendo entre
    # la diferencia entre el máximo y el mínimo, de esta manera no hay pérdida
    # de información
    matriz_normalizada = (matriz_normalizada - minimo) / (maximo)
    
    return matriz_normalizada

def pintaI(imagen):
    matriz_normalizada = normalizarmatriz(imagen)
    mostrarimagen(matriz_normalizada)
    
    
# Creo una matriz aleatoria monobanda
matriz = np.random.rand(500,500)
pintaI(matriz)

# Creo una matriz aleatoria multibanda
matriz = np.random.rand(500,500,3)
pintaI(matriz)

input("\n-------Pulsa una tecla para continuar-------\n")

print("Ejercicio 3")
#Ejercicio 3.- Escribir una función que visualice varias imágenes distintas a la
# vez

################################################################################
def pintMI (vim):
    # Las imagenes que sean monobanda las vamos a convertir en tribanda
    # Para ello vamos a recorrer la lista y vamos a comprobar si alguna imagen
    # es monobanda, si lo es simplemente le vamos a agregar en profundidad dos
    # veces la misma matriz.
    for i in range(len(vim)):
        if (vim[i].ndim != 3 ):
            imagen = vim[i]
            vim[i] = np.dstack((np.copy(imagen), np.copy(imagen)))
            vim[i] = np.dstack((vim[i], np.copy(imagen)))
    
    # Vamos a escalar las imagenes para que todas tengan el mismo tamaño que la
    # primera imagen de la lista
    
    # Guardamos las dimensiones maximas de las fotos (la altura maxima de todas
    # las fotos y la anchura maxima de todas las fotos)
    anchuras = []
    alturas = []
    for i in range(len(vim)):
        alturas.append(vim[i].shape[0])
        anchuras.append(vim[i].shape[1])
    
    # Ahora buscamos la maxima de ambas y esa será las dimensiones de nuestras
    # imagenes
    alt_max = np.max(alturas)
    anch_max = np.max(anchuras)
    dim = (anch_max,alt_max)
    
    # Recorremos la lista y ponemos todas las imagenes con la misma dimensión
    for i in range(len(vim)):
        vim[i] = cv.resize(vim[i],dim)
        
    # Concatenamos las imagenes y las mostramos    
    imagen = cv.hconcat(vim)
    mostrarimagen(imagen)
    
################################################################################

imagenes = []
  
# Leo todas las imagenes para concateneralas todas en una (las leo todas en color)
imagenes.append(leeimagen("./images/messi.jpg",0))
imagenes.append(leeimagen("./images/orapple.jpg",flagColor))
imagenes.append(leeimagen("./images/logoOpenCV.jpg",flagColor))
imagenes.append(leeimagen("./images/dave.jpg",flagColor))

# LLamo a la función para que concatene la lista de imagenes y la muestre
pintMI(imagenes)

input("\n-------Pulsa una tecla para continuar-------\n")

print("Ejercicio 4")

# Escribir una función que modifique el color en la imagen
# de cada uno de los elementos de una lista de coordenadas
# de pixeles

def modificacolor(imagen, coordenadas, color):
    resultado = imagen.copy()
    
    for coordenada in coordenadas:
        x,y = coordenada
        
        print(y, "  " , x)
        resultado[y,x] = color
        
    return resultado


imagen = leeimagen("./images/orapple.jpg",flagColor)

coordenadas = []

centro = imagen.shape[0]/2, imagen.shape[1]/2

for i in range(0,50):
    for j in range(0,50):
       coordenadas.append([i+centro[0],j+centro[1]])
                        
imagen_nueva = modificacolor(imagen,coordenadas,[255,0,0])
mostrarimagen(imagen_nueva)

input("\n-------Pulsa una tecla para continuar-------\n")   
    