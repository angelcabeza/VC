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
    
    kernel_normalizado = np.array(kernel)
    
    if ( func == gaussiana ):
        kernel_normalizado = kernel_normalizado / (np.sum(kernel))
    elif ( func == primeraGaussiana):
        kernel_normalizado = sigma * kernel_normalizado
    elif ( func == segundaGaussiana):
        kernel_normalizado = (sigma**2) * kernel_normalizado
        
    
    return kernel_normalizado


print ("Ejercicio 1A")

# Calculamos las mascaras para tamaño 5
mascara_gaussiana_tam5 = kernelGaussiano1D(tam=5)
mascara_primera_gaussiana_tam5 = kernelGaussiano1D(func=primeraGaussiana,tam=5)
mascara_segunda_gaussiana_tam5 = kernelGaussiano1D(func=segundaGaussiana,tam=5)

# Calculamos las mascaras para tamaño 7
mascara_gaussiana_tam7= kernelGaussiano1D(tam=7)
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

# Ejercicio 1.- Escribir una función que lea el fichero de una imagen y
# permita mostrarla tanto en grises como en color
def leeimagen(filename, flagColor):
    img = cv.imread(filename,flagColor)
    
    return img
    

def mostrarimagen(imagen,titulo=""):
    
    plt.title(titulo)
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

def diferenciaImagenes(imagen1,imagen2):
    
    diferencia_imagenes = imagen1 - imagen2
    diferencia_imagenes = np.abs(diferencia_imagenes)
    intensidad_media1 = np.sum(imagen1) / len(imagen1)
    intensidad_media2 = np.sum(imagen2) / len(imagen2)
    dif_min = np.min(diferencia_imagenes)
    dif_max = np.max(diferencia_imagenes)
    
    pixeles_diff = 0
    pixeles_diff_1 = 0
    pixeles_diff_15 = 0
    
    for i in range(diferencia_imagenes.shape[0]):
        for j in range(diferencia_imagenes.shape[1]):
            if (diferencia_imagenes[i][j] > 0.0001):
                pixeles_diff += 1
        
            if (diferencia_imagenes[i][j] > 1):
                pixeles_diff_1 += 1
        
            if (diferencia_imagenes[i][j] > 15):
                pixeles_diff_15 += 1
            
    
    print("La intensidad media de la primera imagen es {} y de la segunda es {}".format(intensidad_media1,intensidad_media2))
    print("La diferencia mediana en nivel de gris es de: {} ".format(diferencia_imagenes[diferencia_imagenes.shape[0] // 2][diferencia_imagenes.shape[1] // 2]))
    print("La diferencia maxima en nivel de gris es de: {} ".format(dif_max))
    print("La diferencia minima en nivel de gris es de: {} ".format(dif_min))
    print("El numero de pixeles diferentes en la imagen es de: {}".format(pixeles_diff))
    print("El numero de pixeles diferentes con una diferencia > 1 es de: {}".format(pixeles_diff_1))
    print("El numero de pixeles diferentes con una diferencia > 15 es de: {}".format(pixeles_diff_15))
        
    
    
def pintarresultados (imagenes,titulos):
    fig = plt.figure()
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


mascara = np.array([1,2,1])


imagen = [[10.0,20.0,30.0],
          [40.0,50.0,60.0],
          [70.0,80.0,90.0]]
  
imagen = np.array(imagen)

imagen_borde = aniade_bordes(imagen, mascara, cv.BORDER_CONSTANT)

nueva_imagen = convulcionar(imagen_borde, mascara)

print("Solucion matriz de ejemplo: ",nueva_imagen)
print()

bicycle = leeimagen('./imagenes/bicycle.bmp',0).astype(float)
    
bicycle_borde = aniade_bordes(bicycle, mascara_gaussiana_tam9,cv.BORDER_REFLECT)

nueva_imagen = normalizarmatriz(convulcionar(bicycle_borde, mascara_gaussiana_tam9))

sigma = (len(mascara_gaussiana_tam9) - 1) / 6

imagen_cv = normalizarmatriz(cv.GaussianBlur(bicycle,ksize=(9,9),sigmaX=sigma,sigmaY=sigma,borderType=cv.BORDER_REFLECT))

titulos = ["Conv manual", "Conv OpenCV"]
pintarresultados([nueva_imagen, imagen_cv],titulos)

print("Diferencia entre las imagenes con convolución Gaussiana: ")
print()
diferenciaImagenes(nueva_imagen,imagen_cv)

# Calculo de la primera derivada de la bicileta
mascara = np.array(mascara_primera_gaussiana_tam9)

mascara_opencv_1d,_ = cv.getDerivKernels(1,1,9)
mascara_opencv_1d = mascara_opencv_1d.reshape(9,)
bycicle_1_deri = normalizarmatriz(convulcionar(bicycle_borde, mascara))
bycicle_1_deri_opencv = normalizarmatriz(convulcionar(bicycle_borde,mascara_opencv_1d))

#################################################################################


# Calculo de la segunda derivada de la bicileta
mascara_opencv_2d,_ = cv.getDerivKernels(2, 2, 9)
mascara_opencv_2d = mascara_opencv_2d.reshape(9,)

mascara = np.array(mascara_segunda_gaussiana_tam9)
bycicle_2_deri = (convulcionar(bicycle_borde, mascara))
bycicle_2_deri_opencv = (convulcionar(bicycle_borde,mascara_opencv_2d))

##############################################################################

titulos = ["Bicylce 1 derivada", "Bicycle 1 derivada OpenCV"]
pintarresultados([bycicle_1_deri,bycicle_1_deri_opencv],titulos)


titulos = ["Bicycle 2 derivada", "Bicycle 2 derivada OpenCV"]
pintarresultados([bycicle_2_deri,bycicle_2_deri_opencv], titulos)
pintarresultados([bycicle_2_deri], ["Bicycle 2 derivada"])


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
    
cat = leeimagen('./imagenes/cat.bmp', 0).astype(float)


cat_s1 = Laplaciana(cat, sigma=1)

cat_s2 = Laplaciana(cat, sigma=3)

titulos = ['Laplaciana sigma=1','Laplaciana sigma=3']
pintarresultados([cat_s1,cat_s2],titulos)

input("<--- Pulse cualquier tecla para continuar --->")

print("Ejercicio 2")

def pintaMIZP (vim,titulo=""):
    # Almacenamos las alturas de todas las imágenes
    alturas = []
    for i in range(len(vim)):
        alturas.append(vim[i].shape[0])
        
    #Calculamos el máximo de las alturas
    max_alt = np.max(alturas)
    
    #convertimos a tribanda la primera imagen si hace falta
    img_resultado = vim[0]
    
    # si su altura no alcanza la altura maxima entramos
    if img_resultado.shape[0] < max_alt:
        # Calculamos la diferencia entre la altura maxima y la altura de la imágen
        # y la dividimos entre 2 para tener una franja negra arriba y abajo
        # además también almacenamos el resto por si no es divisible entre 2
        anc_franja = (max_alt - vim[0].shape[0])
        
        # creamos las dos franjas con las dimensiones recientemente calculadas
        franja_inf = np.zeros( (anc_franja, vim[0].shape[1]) )
        
        # Y las insertamos en la imagen anterior
        img_resultado = np.vstack((img_resultado, franja_inf))

    # Repetimos el mismo proceso pero para todas las imagenes
    for i in range (1,len(vim)):
        img_zeropad = vim[i]
        
        if vim[i].shape[0] < max_alt:
            anc_franja = (max_alt - vim[i].shape[0]) 
            
            tupla = (anc_franja, vim[i].shape[1])
            franja_inf = np.zeros(tupla)

            img_zeropad = np.vstack((img_zeropad, franja_inf))

        # Vamos añadiendo imágenes horizontalmente para concatenarlas
        img_resultado = np.hstack((img_resultado, img_zeropad ))
        
    # Mostramos la imagen y salimos
    mostrarimagen(img_resultado,titulo)

def apilar_piramide(piramide):

    anchura_primera_gaussiana = piramide[1].shape[1]

    imagen_final = piramide[1]

    # apilamos desde el nivel 1 hasta el N
    for i in range(2, len(piramide)):
        forma = (piramide[i].shape[0], anchura_primera_gaussiana)

        if piramide[0].ndim == 3:
            forma = (piramide[i].shape[0], anchura_primera_gaussiana, 3)

        ajustada = np.ones(forma)

        ajustada[:piramide[i].shape[0], :piramide[i].shape[1]] = piramide[i]

        imagen_final = np.vstack((imagen_final, ajustada))

    # añadimos por la izquierda la imagen original, la base de la piramide
    # teniendo en cuenta si es mas grande la imagen original o la union de los
    # niveles a la hora de unirlas
    if piramide[0].shape[0] > imagen_final.shape[0]:
        forma = (piramide[0].shape[0], imagen_final.shape[1])

        if piramide[0].ndim == 3:
            forma = (piramide[0].shape[0], imagen_final.shape[1], 3)

        ajustada = np.ones(forma)
        ajustada[:imagen_final.shape[0], :imagen_final.shape[1]] = imagen_final
        imagen_final = ajustada

    elif piramide[0].shape[0] < imagen_final.shape[0]:
        forma = (imagen_final.shape[0], piramide[0].shape[1])

        if piramide[0].ndim == 3:
            forma = (imagen_final.shape[0], piramide[0].shape[1], 3)

        ajustada = np.ones(forma)
        ajustada[:piramide[0].shape[0], :piramide[0].shape[1]] = piramide[0]
        piramide[0] = ajustada



    imagen_final = np.hstack((piramide[0], imagen_final))

    return imagen_final
def piramideGaussiana (imagen,sigma,niveles,tipo_borde):
    piramide = []
    piramide.append(imagen)
    
    img_actual = imagen.copy()
    
    mascara = kernelGaussiano1D(sigma=sigma)
    
    for i in range(niveles):
        img_actual = aniade_bordes(img_actual, mascara, tipo_borde)
        img_convol = convulcionar(img_actual, mascara)
        img_actual = img_convol[::2,::2]
        #img_actual = normalizarmatriz(img_actual)
        piramide.append(img_actual)
        
    return piramide

def piramideLaplaciana (imagen,sigma,niveles,tipo_borde):
    piramide = []
    piramide_gauss = piramideGaussiana(imagen, sigma, niveles, tipo_borde)
    
    for i in range(niveles):
        img_gaussiana = piramide_gauss[i+1]
        tam = (piramide_gauss[i].shape[1], piramide_gauss[i].shape[0])
        img_gaussiana = cv.resize(src=img_gaussiana,dsize=tam,interpolation=cv.INTER_LINEAR)
        
        laplaciana = piramide_gauss[i] - img_gaussiana
        
        piramide.append(laplaciana)
        
    
    piramide.append(piramide_gauss[-1])
    
    return piramide
    
    

motocicleta = leeimagen('./imagenes/motorcycle.bmp', 0).astype(float)
piramide = piramideGaussiana(motocicleta,1,4,cv.BORDER_REFLECT)

pintaMIZP(piramide,"Piramide Gaussiana")


input("<--- Pulse cualquier tecla para continuar --->")
print("Ejercicio 2B")

piramideLap = piramideLaplaciana(motocicleta,1,4,cv.BORDER_REFLECT)

for i in range(len(piramideLap)):
    piramideLap[i] = normalizarmatriz(piramideLap[i])
    
imagen = apilar_piramide(piramideLap)
mostrarimagen(imagen,"Piramide Laplaciana")

input("<--- Pulse cualquier tecla para continuar --->")
print("Ejercicio 2C")

def reconstruirImagen(piramide,niveles):
    
    imagen = piramide[-1]
    
    for i in range(niveles,0,-1):
        tam = (piramide[i-1].shape[1], piramide[i-1].shape[0])
        imagen = cv.resize(imagen,dsize=tam,interpolation=cv.INTER_LINEAR)
        imagen = imagen + piramide[i-1]
        
    return imagen


piramideLap = piramideLaplaciana(motocicleta,1,4,cv.BORDER_REFLECT)
imagen = reconstruirImagen(piramideLap,4)

pintarresultados([motocicleta,imagen], ['Motocicleta original','Motocicleta reconstruida'])

print ("Diferencia entre la motocicleta original y la motocicleta reconstruida: ")
print()
diferenciaImagenes(motocicleta,imagen)


input("<--- Pulse cualquier tecla para continuar --->")

##################################################################################
print("Bonus 1A")

img_altas_frecuencias = leeimagen('./imagenes/einstein.bmp', 0)
img_bajas_frecuencias = leeimagen('./imagenes/marilyn.bmp', 0)

img_altas_frecuencias = normalizarmatriz(Laplaciana(img_altas_frecuencias,sigma=2))


mascara_bajas_frecuencias = kernelGaussiano1D(sigma=5)
img_bajas_frecuencias = aniade_bordes(img_bajas_frecuencias, mascara_bajas_frecuencias, cv.BORDER_CONSTANT)
img_bajas_frecuencias = normalizarmatriz(convulcionar(img_bajas_frecuencias, mascara_bajas_frecuencias))

img_hibrida = img_bajas_frecuencias + img_altas_frecuencias


pintarresultados([img_altas_frecuencias,img_bajas_frecuencias,img_hibrida], ['Altas Frecuencias', 'Bajas Frecuencias','Hibrida'])

###############################################################################

input("<--- Pulse cualquier tecla para continuar --->")

print("Bonus 1B")

img_bird_altas = leeimagen('./imagenes/bird.bmp',0)
img_plane_bajas = leeimagen('./imagenes/plane.bmp',0)

img_bird_altas = normalizarmatriz(Laplaciana(img_bird_altas,sigma=3))

mascara_bajas_plane = kernelGaussiano1D(sigma=6)

img_plane_bajas = aniade_bordes(img_plane_bajas, mascara_bajas_plane, cv.BORDER_CONSTANT)
img_plane_bajas = normalizarmatriz(convulcionar(img_plane_bajas, mascara_bajas_plane))

img_hibrida_plane = img_plane_bajas + img_bird_altas

pintarresultados([img_bird_altas,img_plane_bajas,img_hibrida_plane], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

#####################################################################################

img_fish_altas = leeimagen('./imagenes/fish.bmp', 0)
img_submarine_bajas = leeimagen('./imagenes/submarine.bmp', 0)

img_fish_altas = normalizarmatriz(Laplaciana(img_fish_altas,sigma=2))

mascara_bajas_submarine = kernelGaussiano1D(sigma=8)

img_submarine_bajas = aniade_bordes(img_submarine_bajas, mascara_bajas_submarine, cv.BORDER_CONSTANT)
img_submarine_bajas = normalizarmatriz(convulcionar(img_submarine_bajas, mascara_bajas_submarine))

img_hibrida_submarine = img_submarine_bajas + img_fish_altas
 
pintarresultados([img_fish_altas,img_submarine_bajas,img_hibrida_submarine], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

######################################################################################

img_cat_altas = leeimagen('./imagenes/cat.bmp',0)
img_dog_bajas = leeimagen('./imagenes/dog.bmp',0)

img_cat_altas = normalizarmatriz(Laplaciana(img_cat_altas,sigma=1.8))

mascara_bajas_dog = kernelGaussiano1D(sigma=5)

img_dog_bajas = aniade_bordes(img_dog_bajas, mascara_bajas_dog, cv.BORDER_CONSTANT)
img_dog_bajas = normalizarmatriz(convulcionar(img_dog_bajas, mascara_bajas_dog))

img_hibrida_dog = img_dog_bajas + img_cat_altas

pintarresultados([img_cat_altas,img_dog_bajas,img_hibrida_dog], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

input("<--- Pulse cualquier tecla para continuar --->")

###############################################################################

print("Bonus 1C")

# Piramide para Marilyn y Einstein
piramide = piramideGaussiana(img_hibrida, 1, 4, cv.BORDER_CONSTANT)
imagen = apilar_piramide(piramide)
mostrarimagen(imagen,"Piramide Marilyn y Einstein")

# Piramide para Bird y plane
piramide = piramideGaussiana(img_hibrida_plane, 1, 4, cv.BORDER_CONSTANT)
imagen = apilar_piramide(piramide)
mostrarimagen(imagen,"Piramide Plane y Bird")

# Piramide para Fish y Submarine
piramide = piramideGaussiana(img_hibrida_submarine, 1, 4, cv.BORDER_CONSTANT)
imagen = apilar_piramide(piramide)
mostrarimagen(imagen,"Piramide Fish y Submarine")

# Piramide para Dog y Cat
piramide = piramideGaussiana(img_hibrida_dog, 1, 4, cv.BORDER_CONSTANT)
imagen = apilar_piramide(piramide)
mostrarimagen(imagen,"Piramide Dog y Cat")

input("<--- Pulse cualquier tecla para continuar --->")
##############################################################################
print("Bonus 2")

img_altas_frecuencias = leeimagen('./imagenes/einstein.bmp', 1)
img_bajas_frecuencias = leeimagen('./imagenes/marilyn.bmp', 1)

img_altas_frecuencias = normalizarmatriz(Laplaciana(img_altas_frecuencias,sigma=2))


mascara_bajas_frecuencias = kernelGaussiano1D(sigma=5)
img_bajas_frecuencias = aniade_bordes(img_bajas_frecuencias, mascara_bajas_frecuencias, cv.BORDER_CONSTANT)
img_bajas_frecuencias = normalizarmatriz(convulcionar(img_bajas_frecuencias, mascara_bajas_frecuencias))

img_hibrida = img_bajas_frecuencias + img_altas_frecuencias

img_hibrida = normalizarmatriz(img_hibrida)

pintarresultados([img_altas_frecuencias,img_bajas_frecuencias,img_hibrida], ['Altas Frecuencias', 'Bajas Frecuencias','Hibrida'])
print("Clipping en einstein")
################################################################################
img_bird_altas = leeimagen('./imagenes/bird.bmp',1)
img_plane_bajas = leeimagen('./imagenes/plane.bmp',1)

img_bird_altas = normalizarmatriz(Laplaciana(img_bird_altas,sigma=3))

mascara_bajas_plane = kernelGaussiano1D(sigma=6)

img_plane_bajas = aniade_bordes(img_plane_bajas, mascara_bajas_plane, cv.BORDER_CONSTANT)
img_plane_bajas = normalizarmatriz(convulcionar(img_plane_bajas, mascara_bajas_plane))

img_hibrida_plane = img_plane_bajas + img_bird_altas

img_hibridaplane = normalizarmatriz(img_hibrida_plane)

pintarresultados([img_bird_altas,img_plane_bajas,img_hibrida_plane], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

print("Clipping n bird")
#####################################################################################

img_fish_altas = leeimagen('./imagenes/fish.bmp', 1)
img_submarine_bajas = leeimagen('./imagenes/submarine.bmp', 1)

img_fish_altas = normalizarmatriz(Laplaciana(img_fish_altas,sigma=2))

mascara_bajas_submarine = kernelGaussiano1D(sigma=8)

img_submarine_bajas = aniade_bordes(img_submarine_bajas, mascara_bajas_submarine, cv.BORDER_CONSTANT)
img_submarine_bajas = normalizarmatriz(convulcionar(img_submarine_bajas, mascara_bajas_submarine))

img_hibrida_submarine = img_submarine_bajas + img_fish_altas

img_hibrida_submarine = normalizarmatriz(img_hibrida_submarine)
 
pintarresultados([img_fish_altas,img_submarine_bajas,img_hibrida_submarine], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

print("Clipping en submarine")
######################################################################################

img_cat_altas = leeimagen('./imagenes/cat.bmp',1)
img_dog_bajas = leeimagen('./imagenes/dog.bmp',1)

img_cat_altas = normalizarmatriz(Laplaciana(img_cat_altas,sigma=1.8))

mascara_bajas_dog = kernelGaussiano1D(sigma=5)

img_dog_bajas = aniade_bordes(img_dog_bajas, mascara_bajas_dog, cv.BORDER_CONSTANT)
img_dog_bajas = normalizarmatriz(convulcionar(img_dog_bajas, mascara_bajas_dog))

img_hibrida_dog = img_dog_bajas + img_cat_altas

img_hibrida_dog = normalizarmatriz(img_hibrida_dog)

pintarresultados([img_cat_altas,img_dog_bajas,img_hibrida_dog], ['Altas Frecuencias' , 'Bajas Frecuencias', 'Hibrida'])

input("<--- Pulse cualquier tecla para continuar --->")
