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
    return ( ( (x**2 + sigma**2) * math.exp(( x**2 / (2*sigma**2) )) ) / sigma**4 )


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


plt.title("Mascaras 2ª derivada de la gaussiana para tamaño 9")
plt.plot(x,mascara_segunda_gaussiana_tam9,'-or',label='Mascara función casera')
plt.plot(x,mask_2d_tam9_cv,'-ob',label='Máscara función openCV')
plt.legend()
plt.xlabel("Puntos evaluados")
plt.ylabel("Primera derivada de la gaussiana")
plt.show()

input("<--- Pulse cualquier tecla para continuar --->")

print("Ejercicio 1B")


kernel_binomial = np.array([1,2,1])
kernel_lon5 = np.array([1,2,1])
kernel_lon7 = np.array([1,2,1])
kernel_lon5_deriv = np.array([-1,0,1])
kernel_lon7_deriv = np.array([-1,0,1])

kernel_deriv = np.array([-1,0,1])

for i in range(0,1):
    kernel_lon5 = np.convolve(kernel_lon5, kernel_binomial)
    kernel_lon5_deriv = np.convolve(kernel_lon5_deriv, kernel_binomial)

for i in range(0,2):
    kernel_lon7 = np.convolve(kernel_lon7, kernel_binomial)
    kernel_lon7_deriv = np.convolve(kernel_lon7_deriv, kernel_binomial)

print("Kernel alisamiento lon 5: ", kernel_lon5)
print("Kernel alisamiento lon 7: ", kernel_lon7)

print("Kernel derivada lon 5: ", kernel_lon5_deriv)
print("Kernel derivada lon 7: ", kernel_lon7_deriv)


kernel_tam9_opencv,kernel_tam9_deriv_opencv = cv.getDerivKernels(0,1,9)

print("Kernel tam 9 alisamiento por OpenCV: ", np.array(kernel_tam9_opencv).T)
print("Kernel tam 9 derivada por OpenCV: ", np.array(kernel_tam9_deriv_opencv).T)

kernel_lon7 = np.convolve(kernel_lon7,kernel_binomial)
kernel_lon7_deriv = np.convolve(kernel_lon7_deriv, kernel_binomial)

print("Kernel tam 9 alisamiento calculado por mi : ", np.array(kernel_lon7))
print("Kernel tam 9 derivada calculado por mi : ", np.array(kernel_lon7_deriv))

input("<--- Pulse cualquier tecla para continuar --->")

print("Ejercicio 1C")

def aniade_bordes(imagen,mascara,tipo_borde):    
    borde = int( (len(mascara) -1) / 2)     
    imagen_borde = cv.copyMakeBorder(imagen, borde, borde, borde, borde, tipo_borde)

    return imagen_borde



def convulcionar (imagen,kernel_horizontal,kernel_vertical):
    
    ini_imagen = int( (len(kernel_horizontal) - 1) / 2 )
    
    anchura = imagen.shape[0]
    altura = imagen.shape[1]
    
    imagen_convolucion = imagen.copy()

    
    original_shape = (imagen.shape[0]- (ini_imagen*2), imagen.shape[1]-(ini_imagen*2))
    
    imagen_nueva = np.zeros(original_shape)
    
    for i in range(ini_imagen, anchura - ini_imagen):
        for j in range (ini_imagen, altura - ini_imagen):
            imagen_nueva[i-ini_imagen, j-ini_imagen] = np.dot(kernel_horizontal, imagen_convolucion[i, j-ini_imagen:(j+ini_imagen+1)])

    
    imagen_convolucion = aniade_bordes(imagen_nueva,kernel_horizontal,cv.BORDER_CONSTANT)
    
    for i in range(ini_imagen, anchura - ini_imagen):
        for j in range (ini_imagen, altura - ini_imagen):
            imagen_nueva[i-ini_imagen, j-ini_imagen] = np.dot(kernel_vertical, imagen_convolucion[i-ini_imagen:i+ini_imagen+1,j])
    

    return imagen_nueva


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
        
def convertirtribanda (imagen):
    imagen_resultado = imagen.copy()

    # Si la imagen es monobanda la convertimos a tribanda creando dos copias apiladas
    # apiladas de la imagen
    if imagen_resultado.ndim != 3 or imagen_resultado.shape[2] < 3:
        imagen_resultado = np.dstack((imagen_resultado, np.copy(imagen)))
        imagen_resultado = np.dstack((imagen_resultado, np.copy(imagen)))
    
    return imagen_resultado

def normalizarmatriz(imagen):
    # Convertimos la imagen a tribanda si esta no lo es
    img_normalizada = convertirtribanda(imagen).astype(float)
    
    # Normalizamos cada canal de la imagen
    for canal in (0,1,2):
        # Obtenemos el minimo y el maximo de cada canal
        minimo = np.min(img_normalizada.transpose(2,0,1)[canal])
        maximo = np.max(img_normalizada.transpose(2,0,1)[canal])
        
        # Comprobamos si dividimos por 0, si no, normalizamos restando el minimo y diviendiendo entre la
        # diferencia entre maximo y minimo y si sí simplemente restamos el minimo sin dividir
        if maximo - minimo != 0:
            img_normalizada.transpose(2,0,1)[canal] = (img_normalizada.transpose(2,0,1)[canal] - minimo) / (maximo - minimo)
        else:
            img_normalizada.transpose(2,0,1)[canal] = img_normalizada.transpose(2,0,1)[canal] - minimo

    return img_normalizada


def pintaMIZP (vim):
    # Almacenamos las alturas de todas las imágenes
    alturas = []
    for i in range(len(vim)):
        alturas.append(vim[i].shape[0])
        
    #Calculamos el máximo de las alturas
    max_alt = np.max(alturas)
    
    #convertimos a tribanda la primera imagen si hace falta
    img_resultado = convertirtribanda(vim[0])
    
    # si su altura no alcanza la altura maxima entramos
    if img_resultado.shape[0] < max_alt:
        # Calculamos la diferencia entre la altura maxima y la altura de la imágen
        # y la dividimos entre 2 para tener una franja negra arriba y abajo
        # además también almacenamos el resto por si no es divisible entre 2
        anc_franja = (max_alt - vim[0].shape[0]) // 2
        anch_extra = (max_alt - vim[0].shape[0]) % 2
        
        # creamos las dos franjas con las dimensiones recientemente calculadas
        franja_sup = np.zeros( (anc_franja + anch_extra, vim[0].shape[1]) )
        franja_inf = np.zeros( (anc_franja, vim[0].shape[1]) )
        
        # Convertimos las franjas a tribanda
        franja_sup = convertirtribanda(franja_sup)
        franja_inf = convertirtribanda(franja_inf)
        
        # Y las insertamos en la imagen anterior
        im_franja_arriba = np.vstack((franja_sup, img_resultado))
        img_resultado = np.vstack((im_franja_arriba, franja_inf))

    # Repetimos el mismo proceso pero para todas las imagenes
    for i in range (1,len(vim)):
        img_zeropad = convertirtribanda(vim[i])
        
        if vim[i].shape[0] < max_alt:
            anc_franja = (max_alt - vim[i].shape[0]) // 2
            anch_extra = (max_alt - vim[i].shape[0]) % 2
        
            tupla = (anc_franja + anch_extra, vim[i].shape[1])
            franja_sup = np.zeros(tupla)
            tupla = (anc_franja, vim[i].shape[1])
            franja_inf = np.zeros(tupla)
            
            franja_sup = convertirtribanda(franja_sup)
            franja_inf = convertirtribanda(franja_inf)

            im_franja_arriba = np.vstack((franja_sup, img_zeropad))
            img_zeropad = np.vstack((im_franja_arriba, franja_inf))
        
        # Vamos añadiendo imágenes horizontalmente para concatenarlas
        img_resultado = np.hstack((img_resultado, img_zeropad ))
            
    # Mostramos la imagen y salimos
    mostrarimagen(img_resultado)


def pintarresultados (imagenes,titulos):
    fig = plt.figure(figsize=(10,7))

    rows = 2
    columns = 2
    
    
    for i in range(1,(len(imagenes)+1)):
        fig.add_subplot(rows,columns,i)
        plt.imshow(imagenes[i-1],cmap='gray')
        plt.title([titulos[i-1]])
        
    
    
    plt.show()
    
mascara = np.array(mascara_gaussiana_tam9)
bicycle = leeimagen('./imagenes/bicycle.bmp',0)

bicycle_borde = aniade_bordes(bicycle, mascara,cv.BORDER_REFLECT)

nueva_imagen = convulcionar(bicycle_borde, mascara,mascara.T)

imagen_cv = cv.GaussianBlur(bicycle,ksize=(9,9),sigmaX=-1,sigmaY=-1,borderType=cv.BORDER_REFLECT)

titulos = ["Bicycle original", "Bicylce convolucionada manualmente", "Bicycle convolucionada por OpenCv"]
pintarresultados([bicycle,nueva_imagen,imagen_cv],titulos)

mascara = np.array(mascara_primera_gaussiana_tam9)
bycicle_1_deri = convulcionar(bicycle_borde, mascara,mascara.T)
mascara = np.array(mascara_segunda_gaussiana_tam9)
bycicle_2_deri = convulcionar(bicycle_borde, mascara,mascara.T)

titulos = ["Bicycle original", "Bicylce 1 derivada", "Bicycle 2 derivada"]
pintarresultados([bicycle,bycicle_1_deri,bycicle_2_deri],titulos)



###############################################################################

input("<--- Pulse cualquier tecla para continuar --->")


print("Ejercicio 1D")

def Laplaciana(imagen,sigma=None,tam=None):
    imagen_copia = imagen.copy()
    # Comprobamos el dato que nos dan y calculamos
    # el que falte. Si no nos dan ninguno avisamos
    # al usuario de que tiene que pasar al menos 1
    # y paramos el programa
    if (sigma != None):
        mascara_gaussiana = kernelGaussiano1D(sigma=sigma)
        mascara_seg_gaussiana = kernelGaussiano1D(func=segundaGaussiana,sigma=sigma)
    elif (tam != None):
        mascara_gaussiana = kernelGaussiano1D(tam=tam)
        mascara_seg_gaussiana = kernelGaussiano1D(func=segundaGaussiana,tam=tam)
    else:
        assert("Debe pasar un valor de sigma o un tamaño de mascara")

    # Le añado el borde a la imagen para que esta no pierda tamaño
    imagen_borde = aniade_bordes(imagen_copia, mascara_seg_gaussiana, cv.BORDER_REFLECT)

    dxx = convulcionar(imagen_borde,mascara_seg_gaussiana,mascara_gaussiana)
    dyy = convulcionar(imagen_borde, mascara_gaussiana,mascara_seg_gaussiana)
    
    imagen_laplaciana = sigma**2 * (np.array(dxx) + np.array(dyy))

    return imagen_laplaciana


##############################################################################
    
cat = leeimagen('./imagenes/motorcycle.bmp', 0).astype(float)


cat_s1 = Laplaciana(cat, sigma=1)

cat_s2 = Laplaciana(cat, sigma=3)

titulos = ['Cat original','Cat Laplaciana sigma=1','Cat Laplaciana sigma=3']
pintarresultados([cat,cat_s1,cat_s2],titulos)
