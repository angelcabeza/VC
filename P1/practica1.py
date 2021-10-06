#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:40:54 2021

@author: angel
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

##############################################################################
## Ejercicio 1

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
    
    kernel_normalizado = kernel
    
    if ( func == gaussiana ):
        kernel_normalizado = kernel_normalizado / np.sum(kernel)
        
    
    return kernel_normalizado


print ("Ejercicio 1A")

# Calculamos las mascaras para tamaño 5
mascara_gaussiana_tam5 = kernelGaussiano1D(tam=5)
mascara_primera_gaussiana_tam5 = kernelGaussiano1D(func=primeraGaussiana,tam=5)
mascara_segunda_gaussiana_tam5 = kernelGaussiano1D(func=segundaGaussiana,tam=5)

# Calculamos las mascaras para tamaño 7
mascara_gaussiana_tam5 = kernelGaussiano1D(tam=7)
mascara_primera_gaussiana_tam7 = kernelGaussiano1D(func=primeraGaussiana,tam=7)
mascara_segunda_gaussiana_tam7 = kernelGaussiano1D(func=segundaGaussiana,tam=7)

# Calculamos las mascaras para tamaño 9
mascara_gaussiana_tam9 = kernelGaussiano1D(tam=9)
mascara_primera_gaussiana_tam9 = kernelGaussiano1D(func=primeraGaussiana,tam=9)
mascara_segunda_gaussiana_tam9 = kernelGaussiano1D(func=segundaGaussiana,tam=9)


# Calculamos las mascaras para tamaño 5 con getDerivKernels de openCV
#mascara_gaussiana_tam5_opencv = cv.getDerivKernels(0,0,ksize=5)
mask_1d_tam5_cv, trash = cv.getDerivKernels(1,0,ksize=5)
mask_2d_tam5_cv, trash = cv.getDerivKernels(2,0,ksize=5)

# Calculamos las mascaras para tamaño 7 con getDerivKernels de openCV
#mascara_gaussiana_tam7_opencv = cv.getDerivKernels(0,0,ksize=7)
mask_1d_tam7_cv, trash = cv.getDerivKernels(1,0,ksize=7)
mask_2d_tam7_cv, trash  = cv.getDerivKernels(2,0,ksize=7)

# Calculamos las mascaras para tamaño 9 con getDerivKernels de openCV
#mascara_gaussiana_tam9_opencv = cv.getDerivKernels(0,0,ksize=9)
mask_1d_tam9_cv, trash = cv.getDerivKernels(1,0,ksize=9)
mask_2d_tam9_cv, trash  = cv.getDerivKernels(2,0,ksize=9)


# Vamos a pintar las mascaras que obtenemos para ver sus diferencias
x = []
for i in range(-2,3):
    x.append(i)

plt.title("Mascaras 1ª derivada de la gaussiana para tamaño 5")
plt.plot(x,mascara_primera_gaussiana_tam5,'-or',label='Mascara función casera')
plt.plot(x,mask_1d_tam5_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Valor de la primera derivada de la gaussiana")
plt.show()


plt.title("Mascaras 2ª derivada de la gaussiana para tamaño 5")
plt.plot(x,mascara_segunda_gaussiana_tam5,'-or',label='Mascara función casera')
plt.plot(x,mask_2d_tam5_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Valor de la segunda derivada de la gaussiana")
plt.show()

##############################################################################

input("<--- Pulse cualquier tecla para continuar --->")

###############################################################################

x = []
for i in range(-3,4):
    x.append(i)

plt.title("Mascaras 1ª derivada de la gaussiana para tamaño 7")
plt.plot(x,mascara_primera_gaussiana_tam7,'-or',label='Mascara función casera')
plt.plot(x,mask_1d_tam7_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Valor de la primera derivada de la gaussiana")
plt.show()


plt.title("Mascaras 2ª derivada de la gaussiana para tamaño 7")
plt.plot(x,mascara_segunda_gaussiana_tam7,'-or',label='Mascara función casera')
plt.plot(x,mask_2d_tam7_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Valor de la segunda derivada de la gaussiana")
plt.show()

##############################################################################

input("<--- Pulse cualquier tecla para continuar --->")

##############################################################################
x = []
for i in range(-4,5):
    x.append(i)

plt.title("Mascaras 1ª derivada de la gaussiana para tamaño 9")
plt.plot(x,mascara_primera_gaussiana_tam9,'-or',label='Mascara función casera')
plt.plot(x,mask_1d_tam9_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Primera derivada de la gaussiana")
plt.show()


plt.title("Mascaras 2ª derivada de la gaussiana para tamaño 5")
plt.plot(x,mascara_segunda_gaussiana_tam9,'-or',label='Mascara función casera')
plt.plot(x,mask_2d_tam9_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Primera derivada de la gaussiana")
plt.show()

input("<--- Pulse cualquier tecla para continuar --->")

print("Ejercicio 1B")

input("<--- Pulse cualquier tecla para continuar --->")

print("Ejercicio 1C")

def convulcionar (imagen,kernel):
    
