#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:45:17 2021

@author: angel
"""

import math
import cv2 as cv
import matplotlib as plt
import numpy as np

print("Ejercicio 1")

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
    img = cv.imread(filename,flagColor)
    
    return img
    

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
    
    for i in range(1,num_escalas):
        sigma = 1.6 * math.sqrt( 2**((2*i)/3) - 2**((2*(i-1))/3))
        imagen_convol = octava[-1]
        mascara_convol = kernelGaussiano1D(sigma=sigma)
        imagen_convol = aniade_bordes(imagen_convol, mascara_convol, cv.BORDER_REFLECT)
        next_escala = convulcionar(imagen_convol, mascara_convol)
        octava.append(next_escala)
        
    return octava

def calcular_octavas(imagen, num_octavas):
    piramide = []
    imagen = cv.resize(imagen,dsize=(imagen.shape[1]*2,imagen.shape[0]*2),interpolation=cv.INTER_LINEAR)
    
    for i in range(0,num_octavas):
        nueva_octava = calcular_escalas(imagen,5)
        piramide.append(nueva_octava)
        imagen = imagen[::2,::2]
        
    
    return piramide


def espacio_laplaciano(pir_lowe):
    pir_lap = []
    
    for i in range(len(pir_lowe)):
        octava_lap = []
        for j in range(len(pir_lowe[i])-1):
            octava_lap.append(pir_lowe[i][j] - pir_lowe[i][j+1])
            
        pir_lap.append(octava_lap)

    return pir_lap


##############################################################################
imagen = leeimagen('imagenes/Yosemite1.jpg', 0)

pir_lowe = calcular_octavas(imagen, 4)
pir_lap = espacio_laplaciano(pir_lowe)


    