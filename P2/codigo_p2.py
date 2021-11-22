# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:41:21 2021

@author: mario
"""


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
from scipy.interpolate import interp1d
import time

from os import listdir
from os.path import isfile, join


np.random.seed(10)


path_yosemite1 = "imagenes/Yosemite1.jpg"
path_yosemite2 = "imagenes/Yosemite2.jpg"

path_mosaico = "imagenes/"

path_mosaico_1 = path_mosaico + "IMG_20211030_110413_S.jpg"
path_mosaico_2 = path_mosaico + "IMG_20211030_110415_S.jpg"
path_mosaico_3 = path_mosaico + "IMG_20211030_110417_S.jpg"


##################################################################################################
# Funciones auxiliares

""" 
Leer una imagen con OpenCV. Permite leerla en color o en escala de grises;
por defecto la lee en color.
Parametros:
    filename, path de la imagen
    flagColor, 0 para escala de grises y 1 para color. Por defecto a 1.
Retorna:
    imagen, numpy array con el valor de los pixeles de la imagen. Si flagColor=0
            entonces ndim=2, si flagColor=1 entonces ndim=3
"""
def leeimagen(filename, flagColor=1):
    imagen = cv.imread(filename, flagColor)
    imagen = imagen.astype('float64')
    return imagen


def leer_imagenes_directorio(path_dir, flagColor=1):
    lista_archivos = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
    
    lista_im = []
    
    for arch in lista_archivos:
        lista_im.append(leeimagen(path_dir + arch, flagColor))
        
    return lista_im

"""
Mostrar una imagen titulada.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    titulo, cadena de caracteres con el titulo de la imagen
"""
def mostrar_imagen(imagen, titulo=''):
    plt.figure(figsize=(10, 5))
    
    plt.title(titulo)
    
    plt.axis('off')
    
    if imagen.ndim == 3:
        im_normalizada = normalizar_imagen(imagen)
                
        # Si es tribanda hay que rotarla porque OpenCV lee en GBRy matplotlib utiliza RGB
        plt.imshow(im_normalizada[:,:,::-1])
    else:
        # Sino la dibujamos directamente en escala de grises
        plt.imshow(imagen, cmap='gray')
    
    # plt.savefig("imagenes_memoria/ejercicio_1/yosemite1_keypoints", dpi=300, bbox_inches = "tight")
        
    plt.show()


"""
Convertir una imagen a tribanda. Comprueba que no sea tribanda.
Parametros:
    imagen, array numpy con los valores de los pixeles
Retorna:
    im, array numpy con imagen tribanda
"""
def convertir_tribanda (imagen):
    im = imagen.copy()

    # Si la imagen es monobanda la convertimos a tribanda creando dos copias apiladas 
    # apiladas de la imagen
    if im.ndim != 3 or im.shape[2] < 3:
        im = np.dstack((im, np.copy(imagen)))
        im = np.dstack((im, np.copy(imagen)))
    
    return im.astype('float64')


"""
Normalizar una imagen a [0,1].
Parametros:
    imagen, array numpy con los valores de los pixeles
Retorna:
    im, array numpy normalizado
"""
def normalizar_imagen(imagen):
    im = convertir_tribanda(imagen)
        
    # Normalizamos la imagen tribanda
    for canal in (0,1,2):
        # Obtener el minimo y maximo
        minimo = np.min(im.transpose(2,0,1)[canal])
        maximo = np.max(im.transpose(2,0,1)[canal])
        
        if maximo - minimo != 0:
            im.transpose(2,0,1)[canal] = (im.transpose(2,0,1)[canal] - minimo) / (maximo - minimo)
        else:
            im.transpose(2,0,1)[canal] = im.transpose(2,0,1)[canal] - minimo

    return im


"""
Mostrar una imagen tras normalizarla.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    titulo, cadena de caracteres con el titulo de la imagen
"""
def mostrar_im_normalizada(imagen, titulo=""):
    im = normalizar_imagen(imagen)
    
    mostrar_imagen(im, titulo)


"""
Mostrar varias imagenes con sus respectivos titulos en una misma ventana.
Parametros:
    vim, lista de imagenes (numpy arrays)
    titulos, lista de cadenas de caracteres con los titulos de las imagenes
"""
def mostrar_im_en_bloque(vim, titulos, nrows=1, normalizar=False, tam_fig=(10, 5), titulo_fig=None): 
    fig = plt.figure(figsize=tam_fig)
    
    if nrows == 1:
        ncols = len(vim)
    else:
        ncols = math.ceil(len(vim) / nrows)
    
    for i in range(len(vim)):
        imagen = vim[i]
        
        fig.add_subplot(nrows, ncols, i+1)
        
        if imagen.ndim == 3:
            # Si es tribanda hay que rotarla porque OpenCV lee en GBRy matplotlib utiliza RGB
            if normalizar:
                plt.imshow(normalizar_imagen(imagen)[:,:,::-1])
            else:
                plt.imshow(imagen[:,:,::-1])
            
            plt.axis('off')
            plt.title(titulos[i])
        else:
            # Sino la dibujamos directamente en escala de grises
            plt.imshow(imagen, cmap='gray')
            plt.axis('off')
            plt.title(titulos[i])
    
    if not titulo_fig is None:
        fig.suptitle(titulo_fig())
    
    fig.tight_layout()
    
    # plt.savefig("imagenes_memoria/ejercicio_1/escalas_y_octavas", dpi=300, bbox_inches = "tight")
    
    plt.show()



"""
Funcion que une (concatenando) todas las imagenes en una sola.
Si una imagen es más baja que el resto le añade una franja negra para 
convertirlas todas al mismo tamaño.
Parametros:
    vim, lista de imagenes (numpy arrays)
Retorna:
    im_final, numpy array con las imagenes concatenadas
"""
def unir_imagenes(vim):
    num_im = len(vim)
    
    # Obtener la altura maxima de las imagenes
    alturas = []
    min_values = []

    for i in range(0, num_im):
        alturas.append(vim[i].shape[0])
        min_values.append(np.min(vim[i]))
        
    max_altura = max(alturas)
    
    # Obtener el minimo valor de todos los arrays
    min_value = min(min_values)
    
    # Creamos la imagen final con la primera imagen de la lista (convertida a tribanda)
    im_final = convertir_tribanda(vim[0])

    if im_final.shape[0] < max_altura:
        anchura_franja = max_altura - vim[0].shape[0]
        
        # Crear franja
        franja_negra = np.full((anchura_franja, vim[0].shape[1]), min_value)
        
        # Hay que convertir a tribanda tambien la franja negra
        franja_negra= convertir_tribanda(franja_negra)
        
        # Añadir las franja como nueva fila
        im_final = np.vstack((im_final, franja_negra))

    # Añadimos una franja negra a las imagenes con menor altura
    for i in range(1, num_im):
        im = convertir_tribanda(vim[i])

        # Rellenar con filas negras
        if im.shape[0] < max_altura:
            anchura_franja = max_altura - im.shape[0]
            
            # Crear franja
            franja_negra = np.full((anchura_franja, im.shape[1]), min_value)
            
            # Hay que convertir a tribanda tambien la franja negra
            franja_negra = convertir_tribanda(franja_negra)
            
            # Añadir la franja como nueva fila
            im = np.vstack((im, franja_negra))
        
        # Añadir la imagen procesada a la imagen final
        im_final = np.hstack((im_final, im))

    return normalizar_imagen(im_final)


"""
Comparar dos imagenes estadisticamente.
Parametros:
    im1, numpy array con el valor de los pixeles de la primera imagen
    im2, numpy array con el valor de los pixeles de la segunda imagen
"""
def comparar_imagenes(im1, im2, titulo=None):
    print("\n", "*"*40, sep="")
    
    im1 = normalizar_imagen(im1)
    im2 = normalizar_imagen(im2)
    
    if not titulo is None:
        print(titulo, "\n")
    
    if im1.shape != im2.shape:
        print("Las imagenes no tienen el mismo shape")
        print("El tamaño de la primera es", im1.shape)
        print("El tamaño de la segunda es", im2.shape)
    else:
        print("Las imagenes tienen el mismo shape:", im1.shape)
        
        # num_ele_iguales = np.count_nonzero(im1.flatten()==im2.flatten())
        num_ele_iguales = np.sum( abs(im1.flatten() - im2.flatten()) < 1e-7)
        porcentaje_iguales = (num_ele_iguales*100)/im1.size
        print("Elementos iguales: ", num_ele_iguales, " (", "{:.2f}".format(porcentaje_iguales), "%)", sep="")
        
        # mean_diff = abs(np.mean((im1) - (im2)))
        mean_diff = np.mean(np.sqrt(abs(im1**2 - im2**2)))
        print("\nLa diferencia media es:", "{:.3f}".format(mean_diff))
        
        dist_euclidea_media = abs(np.linalg.norm( normalizar_imagen(im1) - normalizar_imagen(im2) ))
        print("\nLa distancia euclidea media es:", "{:.3f}".format(dist_euclidea_media))
    
    if np.array_equal(im1, im2):
        print("Las imagenes son totalmente iguales")
    
    
    min1 = np.min(im1)
    min2 = np.min(im2)

    max1 = np.max(im1)
    max2 = np.max(im2)
    
    mean1 = np.mean(im1)
    mean1 = 0 if mean1 < 1e-10 else mean1
    mean2 = np.mean(im2)
    mean2 = 0 if mean2 < 1e-10 else mean2
            
    std1 = np.std(im1)
    std2 = np.std(im2)
    
    data = {'Imagen 1': [mean1, std1, min1, max1], 'Imagen 2': [mean2, std2, min2, max2]}  
    
    pd.options.display.float_format = "{:.3f}".format
    df = pd.DataFrame(data, index=["mean", "std", "min", "max"])  
    
    print("\n", df, sep="")
    

"""
Comparar las imagenes de dos listas de imagenes. Se compara la diferencia media
de los pixeles de la imagenes y la distancia euclidea media.
Parametros:
    lista1, primera lista de imagenes
    lista2, segunda lista de imagenes
    lista_columnas, titulo de las imagenes
"""
def diferencias_lista_im (lista1, lista2, lista_columnas):
    mean_diff_list = []
    dist_euclidea_media_list = []
    
    for im1, im2 in zip(lista1, lista2):
        mean_diff_list.append(np.mean(np.sqrt(abs(normalizar_imagen(im1)**2 - normalizar_imagen(im2)**2))))
        
        dist_euclidea_media_list.append(abs(np.linalg.norm( normalizar_imagen(im1) - normalizar_imagen(im2) )))
    
    df = pd.DataFrame(columns=lista_columnas,
                      index=["Diferencia media", "Distancia Euclidea Media"])
    
    df.loc["Diferencia media"] = mean_diff_list
    df.loc["Distancia Euclidea Media"] = dist_euclidea_media_list
    
    print("\n", df, sep="")
    

"""
Funcion gaussina de media 0 y desviacion tipica por parametro.
Parametros:
    x, valor sobre el que se calcula la gaussiana
    sigma, desviacion tipica de la gaussiana
Retorna:
    valor de la gaussiana en x
"""
def gaussiana (x, sigma):
    return math.exp(- (x**2) / (2 * sigma**2))


"""
Primera derivada de la funcion gaussina de media 0 y desviacion
tipica por parametro.
Parametros:
    x, valor sobre el que se calcula la gaussiana
    sigma, desviacion tipica de la gaussiana
Retorna:
    valor de la derivada de la gaussiana en x
"""
def gaussiana_der1 (x, sigma):
    return - (x * math.exp(- (x**2) / (2 * sigma**2))) / (sigma**2)


"""
Segunda derivada de la funcion gaussina de media 0 y desviacion
tipica por parametro.
Parametros:
    x, valor sobre el que se calcula la gaussiana
    sigma, desviacion tipica de la gaussiana
Retorna:
    valor de la segunda derivada de la gaussiana en x
"""
def gaussiana_der2 (x, sigma):
    return ( (x**2 - sigma**2) * math.exp(- (x**2) / (2 * sigma**2)) ) / (sigma**4)



def tamanio_sigma(sigma):
    return 6 * sigma + 1

"""
Calcular sigma o tamaño del kernel gaussiano en funcion del tamaño o del sigma
proporcionado, respectivamente.
Parametros:
    sigma, desviacion tipica del kernel gaussiano (opcional)
    tam, tamaño del kernel gaussiano (opcional)
Retorna:
    sigma_final, desviacion tipica final del kernel gaussiano
    tam_final, tamaño final del kernel gaussiano
"""
def obtener_sigma_tam_final (sigma=None, tam=None):
    if sigma is None:
        if tam is None:
            return None
        else:
            sigma_final = (tam - 1) / 6 # porque 2*(3*sigma)+1=tam
            tam_final = tam
    else:
        tam_final = int(np.ceil(tamanio_sigma(sigma)) // 2 * 2 + 1) # Aproximarlo al entero impar mas cercano
        sigma_final = sigma
    
    return sigma_final, tam_final


"""
Crear una mascara de kernel gaussiano. Se puede especificar
tanto el tamaño de la mascara como la desviacion tipica del kernel.
Tiene preferencia la desviacion tipica sobre el tamaño: si se especifican
los dos parametros pero no coinciden en valor se utiliza el de la 
desviacion tipica.
Debe ser proporcionado al menos un parametro entre sigma y tam.
Se puede escoger el tipo de kernel mediante el parametro flag:
    0 para kernel gaussiano
    1 para kernel con primera derivada del gaussiano
    2 para kernel con segunda derivada del gaussiano
Parametros:
    sigma, desviacion tipica del kernel gaussiano (opcional)
    tam, tamaño del kernel gaussiano (opcional)
    flag, especificar el tipo de kernel a usar (opcional: por defecto kernel gaussiano)
Retorna:
    mask, la mascara del kernel
"""
def kernel_gaussiano (sigma=None, tam=None, flag=0):
    sigma, tam = obtener_sigma_tam_final(sigma, tam)
    
    # Comprobrar que los parametros son correctos
    if sigma is None or tam is None or flag < 0 or flag > 2:
        return None
    
    k_value = int(tam / 2)
    
    mask = np.zeros(tam)
        
    # Calcular la mascara valor a valor
    for i in range(tam):
        if flag == 0:
            mask[i] = gaussiana (-k_value+i, sigma)
        elif flag == 1:
            mask[i] = gaussiana_der1 (-k_value+i, sigma)
        elif flag == 2:
            mask[i] = gaussiana_der2 (-k_value+i, sigma)
    
    mask *= sigma**flag
    
    # El kernel gaussiano tiene que sumar uno
    if flag == 0:
        mask /= np.sum(mask)
    
    return mask


"""
Comparar las derivadas del kernel Gaussiano con los kernels obtenidos con
la funcion getDerivKernels de CV.
Parametros:
    tams, lista de tamaños de los kernels a comparar
"""
def graficos_comparar_kernels (tams):
    for t in tams:
        # Obtener el kernel de la gaussiana
        gauss_mask = kernel_gaussiano(tam=t, flag=0)
        
        # Obtener el kernel de la gaussiana con getDerivKernels
        gDK_x_mask, gDK_y_mask = cv.getDerivKernels(0, 1, t, normalize=True)
        
        # Obtener los kernels derivados de la gaussiana
        gauss_der1_mask = kernel_gaussiano(tam=t, flag=1)
        gauss_der2_mask = kernel_gaussiano(tam=t, flag=2)
        
        # Obtener los kernels derivados con getDerivKernels
        gDK_der1x_mask, cv_der1y_mask = cv.getDerivKernels(1, 0, t, normalize=True)
        dDK_der2x_mask, cv_der2y_mask = cv.getDerivKernels(2, 0, t, normalize=True)
        
        # Espacio de valores eje de abscisas
        x_values = np.arange(int(-t/2), int(t/2)+1)
        
        
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
        
        
        # Primer grafico: 1 gaussiana
        ax[0,0].axhline(y=0, color='black', lw=0.5)
        ax[0,0].axvline(x=0, color='black', lw=0.5)
        ax[0,0].set_title("Gaussiana tam = " + str(t))
        ax[0,0].set_xlabel("X values")
        ax[0,0].set_ylabel("Kernel values")
        
        # Mostrar los puntos del kernel gaussiano
        ax[0,0].scatter(x_values, gauss_mask, color="black")
        
        # Crear una funcion interpolada cuadratica para unir los puntos en el grafico
        f = interp1d(x_values, gauss_mask, kind='quadratic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[0,0].plot (x_new, y_smooth)
        
        
        # Segundo grafico: gaussiana con getDerivKernels
        ax[1,0].axhline(y=0, color='black', lw=0.5)
        ax[1,0].axvline(x=0, color='black', lw=0.5)
        ax[1,0].set_title("GetDerivKernels Gaussiana tam = " + str(t))
        ax[1,0].set_xlabel("X values")
        ax[1,0].set_ylabel("Kernel values")
        
        # Mostrar los puntos de la gaussiana de getDerivKernels
        ax[1,0].scatter(x_values, gDK_x_mask, color="black")
        
        # Crear una funcion interpolada cuadratica para unir los puntos en el grafico
        f = interp1d(x_values, gDK_x_mask.reshape(t,), kind='quadratic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[1,0].plot(x_new, y_smooth, color="red")
        
        
        # Tercer grafico: 1 derivada gaussiana
        ax[0,1].axhline(y=0, color='black', lw=0.5)
        ax[0,1].axvline(x=0, color='black', lw=0.5)
        ax[0,1].set_title("Gaussiana Derivada 1 tam = " + str(t))
        ax[0,1].set_xlabel("X values")
        ax[0,1].set_ylabel("Kernel values")
        
        # Mostrar los puntos de la derivada 1 del kernel gaussiano
        ax[0,1].scatter(x_values, gauss_der1_mask, color="black")
        
        # Crear una funcion interpolada cuadratica para unir los puntos en el grafico
        f = interp1d(x_values, gauss_der1_mask, kind='quadratic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[0,1].plot (x_new, y_smooth)
        
        
        # Cuarto grafico: 1 derivada getDerivKernels
        ax[1,1].axhline(y=0, color='black', lw=0.5)
        ax[1,1].axvline(x=0, color='black', lw=0.5)
        ax[1,1].set_title("GetDerivKernels Derivada 1 tam = " + str(t))
        ax[1,1].set_xlabel("X values")
        ax[1,1].set_ylabel("Kernel values")
        
        # Mostrar los puntos de la derivada 1 de getDerivKernels
        ax[1,1].scatter(x_values, gDK_der1x_mask, color="black")
        
        # Crear una funcion interpolada cuadratica para unir los puntos en el grafico
        f = interp1d(x_values, gDK_der1x_mask.reshape(t,), kind='quadratic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[1,1].plot(x_new, y_smooth, color="red")
        
        
        # Quinto grafico: 2 derivada gaussiana
        ax[0,2].axhline(y=0, color='black', lw=0.5)
        ax[0,2].axvline(x=0, color='black', lw=0.5)
        ax[0,2].set_title("Gaussiana Derivada 2 tam = " + str(t))
        ax[0,2].set_xlabel("X values")
        ax[0,2].set_ylabel("Kernel values")
                
        # Mostrar los puntos de la derivada 2 del kernel gaussiano
        ax[0,2].scatter(x_values, gauss_der2_mask, color="black")
        
        # Crear una funcion interpolada cubica para unir los puntos en el grafico
        f = interp1d(x_values, gauss_der2_mask, kind='cubic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[0,2].plot (x_new, y_smooth)
        
        
        # Sexto grafico: 2 derivada getDerivKernels
        ax[1,2].axhline(y=0, color='black', lw=0.5)
        ax[1,2].axvline(x=0, color='black', lw=0.5)
        ax[1,2].set_title("GetDerivKernels Derivada 2 tam = " + str(t))
        ax[1,2].set_xlabel("X values")
        ax[1,2].set_ylabel("Kernel values")
        
        # Mostrar los puntos de la derivada 2 de getDerivKernels
        ax[1,2].scatter(x_values, dDK_der2x_mask, color="black")
        
        # Crear una funcion interpolada cubica para unir los puntos en el grafico
        f = interp1d(x_values, dDK_der2x_mask.reshape(t,), kind='cubic')
        x_new = np.linspace(x_values.min(), x_values.max(),500)
        y_smooth=f(x_new)
        ax[1,2].plot (x_new, y_smooth, color="red")
        
        
        fig.tight_layout()
    
    plt.show()
        
        

"""
Representar las funciones gaussiana y derivadas.
Parametros:
    sigma, desviacion tipica de los kernels gaussianos a generar.
"""
def grafica_gaussianas(sigma):
    # Crear un espacio de valores para mostrar las funciones
    x_values = np.linspace(-5, 5, 100)
    
    # Generar valores de las gaussianas y sus derivadas
    gauss_values = [gaussiana(x, sigma) for x in x_values]
    gauss_der1_values = [gaussiana_der1(x, sigma) for x in x_values]
    gauss_der2_values = [gaussiana_der2(x, sigma) for x in x_values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Mostrar las funciones gaussianas y derivadas
    ax.plot(x_values, gauss_values, color="red", label="Gaussina")
    ax.plot(x_values, gauss_der1_values, color="blue", label="Gaussiana Derivada 1")
    ax.plot(x_values, gauss_der2_values, color="green", label="Gaussiana Derivada 2")
    
    ax.set_xlabel("X values")
    ax.set_ylabel("Functions values")
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axvline(x=0, color='black', lw=0.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-5, 5)

    ax.legend(loc="upper right")
    
    ax.set_title("Comparacion de funciones gaussiana y derivadas")
        
    plt.show()


"""
Aplicar padding sobre la imagen, creando bordes en esta.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    padding, indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
    width_top, anchura del borde superior
    width_bottom (opcional), anchura del borde inferior. Si no se especifica se toma la misma
        que en el borde superior
    width_left (opcional), anchura del borde de la izquierda. Si no se especifica se toma la misma
        que en el borde superior
    width_right (opcional), anchura del borde de la derecha. Si no se especifica se toma la misma
        que en el borde superior
    cte_padding (opcional), constante para el tipo de borde constante
"""
def aplicar_padding(imagen, padding, width_top, width_bottom=None, width_left=None,
                    width_right=None, cte_padding = 0):
    
    if width_bottom is None:
        width_bottom = width_top
    
    if width_left is None:
        width_left = width_top
    
    if width_right is None:
        width_right = width_top
    
    return cv.copyMakeBorder(imagen, width_top, width_bottom, width_left,
                             width_right, padding, value = cte_padding)


"""
Aplicar una convolucion 1D de un kernel sobre una imagen (matriz).
El tamaño de la imagen se vera reducido: se asume que se ha realiado un padding previamente.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    mask_ver, numpy array con el kernel 1D a aplicar verticalmente
Retorna:
    im_final, numpy array con la imagen convolucionada
"""
def convolve1D(imagen, mask_ver):
    mask_repeated = (np.tile(mask_ver,(imagen.shape[1],1))).T

    # Calcular dimensiones de la imagen (si esta en gris la dimension es 1)
    n_dim = 3 if imagen.ndim == 3 else 1
    
    tam_borde = len(mask_ver) // 2
    
    # Crear la imagen donde se va a ir almacenando la convolucion
    shape_final = list(imagen.shape)
    shape_final[0] -= tam_borde * 2
    im_final = np.zeros(tuple(shape_final))
    
    for dim in range(n_dim):
        for fila in range(tam_borde, imagen.shape[0]-tam_borde):
            if n_dim == 3:
                im_final[fila-tam_borde, :, dim] = np.sum(np.multiply(imagen[fila-tam_borde:fila+tam_borde+1, :, dim],
                                                                      mask_repeated),
                                                          axis=0)
            else:
                im_final[fila-tam_borde, :] = np.sum(np.multiply(imagen[fila-tam_borde:fila+tam_borde+1,:],
                                                                 mask_repeated),
                                                     axis=0)
    
    return im_final


"""
Aplicar una convolucion de un kernel sobre una imagen (matriz). Se aplica con kernels 1D
en vez de 2D por eficiencia en tiempos de ejecucion.
Un kernel 2D debe ser separado (antes de llamar a la funcion) en los respectivos
kernels 1D para aplicar primero la convolucion vertical y luego la horizontal.
El tamaño de la imagen se vera reducido: se asume que se ha realiado un padding previamente.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    mask_ver, numpy array con el kernel 1D a aplicar verticalmente
    mask_hor (opcional), numpy array con el kernel 1D a aplicar horizontalmente.
        Si no se especifica se toma mask_ver como mask_hor
    normalize (opcional), normalizar la imagen final
Retorna:
    im_final, numpy array con la imagen convolucionada
"""
def convolve2D(imagen, mask_ver, mask_hor=None, normalize=False):
    if mask_hor is None:
        mask_hor = mask_ver.copy()
    
    n_dim = 3 if imagen.ndim == 3 else 1
    
    im_intermedia = convolve1D(imagen, mask_ver)
    
    if n_dim== 3:
        im_intermedia = im_intermedia.transpose(1, 0, 2)
    else:
        im_intermedia = im_intermedia.T
    
    im_final = convolve1D(im_intermedia, mask_hor)
    
    if n_dim == 3:
        im_final = im_final.transpose(1, 0, 2)
    else:
        im_final = im_final.T
    
    if normalize:
        im_final = normalizar_imagen(im_final)
    
    return im_final


# """
# Aplicar una convolucion de un kernel sobre una imagen (matriz). Se aplica con kernels 1D
# en vez de 2D por eficiencia en tiempos de ejecucion.
# Un kernel 2D debe ser separado (antes de llamar a la funcion) en los respectivos
# kernels 1D para aplicar primero la convolucion vertical y luego la horizontal.
# El tamaño de la imagen se vera reducido: se asume que se ha realiado un padding previamente.
# Parametros:
#     imagen, numpy array con el valor de los pixeles de la imagen
#     mask_ver, numpy array con el kernel 1D a aplicar verticalmente
#     mask_hor (opcional), numpy array con el kernel 1D a aplicar horizontalmente.
#         Si no se especifica se toma mask_ver como mask_hor
#     normalize (opcional), normalizar la imagen final
# Retorna:
#     im_final, numpy array con la imagen convolucionada
# """
# def convolve2D(imagen, mask_ver, mask_hor=None, normalize=False):    
#     if mask_hor is None:
#         mask_hor = mask_ver.copy()
    
#     bordes_hor = len(mask_ver) // 2 # Anchura bordes horizontales (arriba y abajo)
#     bordes_ver = len(mask_hor) // 2 # Anchura bordes verticales (izquierda y derecha)
    
#     # Calcular anchura y altura de la matriz
#     alt_matriz  = imagen.shape[0]
#     anch_matriz = imagen.shape[1]
    
#     # Calcular dimensiones de la imagen (si esta en gris la dimension es 1)
#     n_dim = 3 if imagen.ndim == 3 else 1
    
#     # Crear la imagen donde se va a ir almacenando la convolucion vertical
#     # Esta imagen intermedia tiene los bordes verticales (derecha e izquierda)
#     # pero no los horizontales (arriba y abajo)
#     shape_intermedia = list(imagen.shape)
#     shape_intermedia[0] -= bordes_hor * 2
#     im_intermedia = np.zeros(tuple(shape_intermedia))
    
#     # Copiar los bordes verticales a la imagen intermedia
#     im_intermedia[:, 0:bordes_ver] = imagen[bordes_hor:alt_matriz-bordes_hor, 0:bordes_ver].copy()
#     im_intermedia[:, -bordes_ver:] = imagen[bordes_hor:alt_matriz-bordes_hor, -1-bordes_ver:-1].copy()
    
#     # Aplicar la convolucion de la mascara vertical a cada dimension de la imagen
#     for dim in range(n_dim):
        
#         # Aplicar a todas las filas que no son bordes horizontales
#         for fil in range(bordes_hor, alt_matriz-bordes_hor):
            
#             # Aplicar a todas las columnas que no son bordes verticales
#             for col in range(bordes_ver, anch_matriz-bordes_ver):
                
#                 # Hay que diferenciar entre monobanda y tribanda
#                 if n_dim == 3:
#                     im_intermedia[fil-bordes_hor, col, dim] = np.dot(mask_ver,
#                                                                      imagen[fil-bordes_hor:fil+bordes_hor+1, col, dim])
#                 else:
#                     im_intermedia[fil-bordes_hor, col] = np.dot(mask_ver,
#                                                                 imagen[fil-bordes_hor:fil+bordes_hor+1, col])
    
    
#     # Crear la imagen final donde se va a ir almacenando la convolucion horizontal
#     shape_final = shape_intermedia.copy()
#     shape_final[1] -= bordes_ver * 2
#     im_final = np.zeros(tuple(shape_final))
    
#     # Aplicar la convolucion de la mascara horizontal a cada dimension de la imagen
#     for dim in range(n_dim):
        
#         # Aplicar a todas las filas horizontales de la imagen intermedia
#         for fil in range(0, im_intermedia.shape[0]):
            
#             # Aplicar a todas las columnas que no son bordes verticales
#             for col in range(bordes_ver, anch_matriz-bordes_ver):
                
#                 # Hay que diferenciar entre monobanda y tribanda
#                 if n_dim == 3:
#                     im_final[fil, col-bordes_ver, dim] = np.dot(mask_hor,
#                                                                 im_intermedia[fil, col-bordes_ver:col+bordes_ver+1, dim])
#                 else:
#                     im_final[fil, col-bordes_ver] = np.dot(mask_hor,
#                                                            im_intermedia[fil, col-bordes_ver:col+bordes_ver+1])
    
#     if normalize:
#         im_final = normalizar_imagen(im_final)
    
#     return im_final


"""
Aplicar la Laplaciana sobre una imagen.
El procedimiento para calcularlo es el siguiente:
    i. Calcular dxx: convolucion por filas de la derivada segunda de la gaussiana y por 
                     columnas de la gaussiana.
    ii. Calcular dyy: convolucion por filas de la gaussiana y por columnas de la
                      derivada de la gaussiana.
    iii. Sumar dxx y dyy.
    iv. No es necesario multiplicar el resultado por sigma^2 porque los kernels de la
        segunda derivada ya estan normalizados.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    sigma, desviacion tipica del kernel
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - valor por defecto del parametro
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    im_final, numpy array con la imagen convolucionada
"""
def laplacianConvolution(imagen, sigma, padding=cv.BORDER_REFLECT):
    # Obtener kernel gaussiano
    mask_gauss = kernel_gaussiano(sigma, flag=0)
    # Obtener kernel derivada 2 gaussiana
    mask_gauss_der2 = kernel_gaussiano(sigma, flag=2)
    
    tam_kernel = len(mask_gauss)
    
    # Aplicar padding a la imagen
    im_padding = aplicar_padding(imagen, padding, tam_kernel//2)
    
    # Calcular las derivadas 2 de la imagen respecto a x y respecto a y
    imagen_dxx = convolve2D(im_padding, mask_gauss_der2, mask_gauss)
    imagen_dyy = convolve2D(im_padding, mask_gauss, mask_gauss_der2)
    
    return imagen_dxx + imagen_dyy



def imagen_derivadas_vs_laplaciana(imagen, sigma, padding=cv.BORDER_REFLECT):
    im_laplaciana = laplacianConvolution(imagen, sigma, padding)
    
    mask_gauss = kernel_gaussiano(sigma, flag=0)
    mask_gauss_der2 = kernel_gaussiano(sigma, flag=2)
    
    imagen_dxx = convolve2D(imagen, mask_gauss, mask_gauss_der2)
    imagen_dyy = convolve2D(imagen, mask_gauss_der2, mask_gauss)
    
    mostrar_im_en_bloque([imagen, imagen_dxx, imagen_dyy, im_laplaciana],
                         ["Original", "Segunda derivada en x",
                          "Segunda derivada en y", "Imagen laplaciana"],
                         nrows=2)


"""
Aplicar los kernels gaussiano y sus derivados sobre una imagen.
La funcion realiza el preprocesado de la imagen añadiendole padding.
Debe ser proporcionado al menos un parametro entre sigma y tam.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    sigma (opcional), desviacion tipica de las funciones gaussianas y derivadas
    tam (opcional), tamaño de los kernels
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - valor por defecto del parametro
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
"""
def aplicar_gaussianas(imagen, sigma=None, tam=None, padding=cv.BORDER_REFLECT):
    # Obtener mascaras
    mask_gauss = kernel_gaussiano(sigma, tam, 0)
    mask_gauss_der1 = kernel_gaussiano(sigma, tam, 1)
    mask_gauss_der2 = kernel_gaussiano(sigma, tam, 2)
    
    # Añadir padding a la imagen
    im = aplicar_padding(imagen, padding, width_top=len(mask_gauss//2))
    
    # Aplicar convoluciones
    imagen_gauss = convolve2D(im, mask_gauss)
    imagen_gauss_der1 = convolve2D(im, mask_gauss_der1)
    imagen_gauss_der2 = convolve2D(im, mask_gauss_der2)
        
    lista_im = [imagen, imagen_gauss, imagen_gauss_der1, imagen_gauss_der2]
    lista_titulos = ["Original", "Alisada", "Derivada 1", "Derivada 2"]
    
    mostrar_im_en_bloque(lista_im, lista_titulos, nrows=2)
    


"""
Comparar la convolucion con el kernel gaussiano implementado en esta practica
con la convolucion gaussiana de OpenCV.
Debe ser proporcionado al menos un parametro entre sigma y tam.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    sigma (opcional), desviacion tipica de las funciones gaussianas y derivadas
    tam (opcional), tamaño de los kernels
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - valor por defecto del parametro
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
"""
def comparar_gaussina_GaussianBlur (imagen, sigma=None, tam=None, padding=cv.BORDER_REFLECT): 
    sigma, tam = obtener_sigma_tam_final(sigma, tam)
    
    mask_gaussiana = kernel_gaussiano(sigma, tam, 0)
    
    # Añadir padding a la imagen
    imagen_padding = aplicar_padding(imagen, padding, width_top=len(mask_gaussiana)//2)
    
    # Aplicar convolucion
    imagen_gaussiana = convolve2D(imagen_padding, mask_gaussiana)
    
    # Aplicar filtro gaussiano con OpenCV
    im_gaussianBlur = cv.GaussianBlur(imagen, (tam, tam), sigmaX=sigma, borderType=padding)
    
    
    lista_im = [imagen, imagen_gaussiana, im_gaussianBlur]
    lista_titulos = ["Original", "Filtro Gaussiano", "Gaussian Blur"]
    
    mostrar_im_en_bloque(lista_im, lista_titulos, nrows=2)
    
    comparar_imagenes(imagen_gaussiana, im_gaussianBlur,
                      "Comparacion alisamiento con filtro gaussiano y GaussianBlur de OpenCV")


"""
Reducir el tamaño de una imagen a la mitad mediante submuestreo.
Se eliminan las filas y columnas impares.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
Retorna:
    im_sin_impares, imagen sin las filas impares y sin las columnas impares
"""
def submuestrear_imagen(imagen):
    # Eliminar filas impares
    im_sin_filas_impares = imagen[::2,:].copy()
    
    # Eliminar columnas impares sobre el anterior resultado
    im_sin_impares = im_sin_filas_impares[:,::2]
    
    return im_sin_impares


"""
Obtener las imagenes reducidas y difuminadas de la piramide gaussiana.
La primera imagen es la imagen original: para completar los niveles de la piramide,
para cada nivel, se aplica un filtro gaussiano a la imagen del nivel anterior y se 
submuestrea la imagen.
El submuestreo se realiza eliminando las filas y columnas impares.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    sigma (opcional), desviacion tipica del kernel gaussiano utilizado para difuminar
    tam (opcional), tamaño del kernel
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    lista_im, lista con las imagenes de la piramide gaussiana
"""
def lista_piramida_gaussiana(imagen, niveles, sigma=None, tam=None, padding=cv.BORDER_REFLECT):
    im = imagen.copy()
    
    # Lista de imagenes de la piramide: la primera imagen (nivel 0) es la imagen original
    lista_im = [im]
    
    # Obtener kernel gaussiano de alisamiento
    mask_gaussiana = kernel_gaussiano(sigma, tam, flag=0)
    
    # Para cada nivel hay que generar una imagen alisada y reducida de tamaño
    # En la iteracion i del bucle se genera la imagen del nivel i+1
    for i in range(niveles):
        # Tomar la imagen del nivel superior y añadirle padding
        im_padding = aplicar_padding(lista_im[i], padding, len(mask_gaussiana)//2)
        
        # Alisar la imagen convolucionandola con el kernel gaussiano
        im_conv = convolve2D(im_padding, mask_gaussiana)
        
        # Submuestrear la imagen eliminando filas y columnas impares
        im_submuestreo = submuestrear_imagen(im_conv)
        
        lista_im.append(im_submuestreo)
    
    return lista_im


"""
Obtener la imagen de la piramide gaussiana. Esta formada por todas las imagenes de la piramide
concatenadas en una sola. La primera imagen es la imagen original: para completar los niveles
de la piramide, para cada nivel, se aplica un filtro gaussiano a la imagen del nivel anterior
y se submuestrea la imagen. El submuestreo se realiza eliminando las filas y columnas impares.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    sigma (opcional), desviacion tipica del kernel gaussiano utilizado para difuminar
    tam (opcional), tamaño del kernel
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    imagen (np array) con la concatenacion de las imagenes de la piramide
"""
def construir_piramide_gaussiana(imagen, niveles, sigma=None, tam=None, padding=cv.BORDER_REFLECT):
    return unir_imagenes( lista_piramida_gaussiana(imagen, niveles, sigma, tam, padding) )


"""
Construir las imagenes de la piramide gaussiana con las funciones de OpenCV.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    lista_im, lista con las imagenes de la piramide gaussiana
"""
def lista_piramida_gaussiana_opencv(imagen, niveles, padding=cv.BORDER_REFLECT):
    im = imagen.copy()
        
    # Lista de imagenes de la piramide: la primera imagen (nivel 0) es la imagen original
    lista_im = [im]
    
    # Para cada nivel de la piramide se genera la imagen alisada y de tamaño reducido
    for i in range(niveles):
        # Aplicar gaussianBlur
        # im_gaussianBlur = cv.GaussianBlur(lista_im[i], (tam, tam), sigmaX=sigma, borderType=padding)
        
        # pyrDown es la funcion de OpenCV que automatiza este proceso
        next_im = cv.pyrDown(lista_im[i], borderType=padding)
        
        lista_im.append(next_im)
    
    return lista_im


"""
Construir la piramide gaussiana con las funciones de OpenCV.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    imagen (np array) con la concatenacion de las imagenes de la piramide
"""
def piramide_gaussiana_OpenCV(imagen, niveles, padding=cv.BORDER_REFLECT):
    return unir_imagenes( lista_piramida_gaussiana_opencv(imagen, niveles, padding) )


"""
Obtener las imagenes de la piramide laplaciana a partir de las imagenes de la piramide gaussiana.
Para obtener la imagen del nivel i:
    1) Aumentar la imagen del nivel i+1 de la piramide gaussiana. El metodo de upsampling utilizado
            es interpolacion lineal (mediante la implementacion de OpenCV).
    2) Obtener la diferencia de la imagen del nivel i de la piramide gaussiana y la imagen aumentada.
La ultima imagen de la piramide laplaciana es igual a la ultima de la gaussiana.
Se utiliza interpolacion lineal (implementacion OpenCV) para aumentar la imagen.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    sigma (opcional), desviacion tipica del kernel gaussiano utilizado para difuminar
    tam (opcional), tamaño del kernel
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    lista_im, lista con las imagenes de la piramide laplaciana
"""
def lista_piramida_laplaciana(imagen, niveles, sigma=None, tam=None, padding=cv.BORDER_REFLECT):
    # Obtener la lista de imagenes de la piramide gaussiana
    lista_gaussiana = lista_piramida_gaussiana(imagen, niveles, sigma, tam, padding)
    
    # Lista de imagenes de la piramide laplaciana
    lista_im = []
    
    # El bucle genera las imagenes de la laplaciana a excepcion de la ultima
    for i in range(1, niveles+1):
        # Obtener el tamaño de la imagen del nivel anterior (i-1) de la piramide gaussiana
        dim = lista_gaussiana[i-1].shape
        
        # Aumentar (upsample) la imagen del nivel actual (i) de la piramide gaussiana
        im_resized = cv.resize(lista_gaussiana[i], (dim[1], dim[0]), interpolation=cv.INTER_LINEAR)
        
        # Generar la imagen del nivel i-1 de la piramide laplaciana: es la diferencia
        # entre la imagen del nivel i-1 en la piramide gaussiana y la imagen aumentada
        im_laplaciana = lista_gaussiana[i-1] - im_resized
        
        lista_im.append(im_laplaciana)
    
    # La imagen del ultimo nivel de la piramide laplaciana es igual a la imagen del ultimo
    # nivel de la piramide gaussiana
    lista_im.append(lista_gaussiana[-1].copy())
    
    return lista_im


"""
Obtener la imagen de la piramide laplaciana. Esta formada por todas las imagenes de la piramide
concatenadas en una sola.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    sigma (opcional), desviacion tipica del kernel gaussiano utilizado para difuminar
    tam (opcional), tamaño del kernel
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    imagen (np array) con la concatenacion de las imagenes de la piramide
"""
def construir_piramide_laplaciana(imagen, niveles, sigma=None, tam=None, padding=cv.BORDER_REFLECT):
    return unir_imagenes( lista_piramida_laplaciana(imagen, niveles, sigma, tam, padding) )


"""
Construir las imagenes de la piramide laplaciana con las funciones de OpenCV.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    lista_im, lista con las imagenes de la piramide laplaciana
"""
def lista_piramida_laplaciana_opencv(imagen, niveles, padding=cv.BORDER_REFLECT):
    # Obtener la lista de imagenes de la piramide gaussiana obtenida con OpenCV
    lista_gaussiana = lista_piramida_gaussiana_opencv(imagen, niveles, padding)
    
    # Lista de imagenes de la piramide laplaciana
    lista_im = []
    
    # El bucle genera las imagenes de la laplaciana a excepcion de la ultima
    for i in range(1, niveles+1):
        # Obtener el tamaño de la imagen del nivel anterior (i-1) de la piramide gaussiana
        dim = lista_gaussiana[i-1].shape
        
        # Aumentar (upsample) la imagen del nivel actual (i) de la piramide gaussiana
        # Se utiliza el metodo pyrUp (metodo opuesto de pyrDown) de OpenCV
        im_resized = cv.pyrUp(lista_gaussiana[i], dstsize=(dim[1], dim[0]))
        
        # Generar la imagen del nivel i-1 de la piramide laplaciana: es la diferencia
        # entre la imagen del nivel i-1 en la piramide gaussiana y la imagen aumentada
        im_laplaciana = cv.subtract(lista_gaussiana[i-1], im_resized)
        
        lista_im.append(im_laplaciana)
    
    # La imagen del ultimo nivel de la piramide laplaciana es igual a la imagen del ultimo
    # nivel de la piramide gaussiana
    lista_im.append(lista_gaussiana[-1])
    
    return lista_im
    

"""
Construir la piramide laplaciana con las funciones de OpenCV.
Parametros:
    imagen, numpy array con el valor de los pixeles de la imagen
    niveles, nº de niveles de la piramide. Cada nivel es una imagen submuestreada
    padding (opcional), indicador del tipo de padding a aplicar:
        cv.BORDER_CONSTANT: añadir un valor constante - padding por defecto
        cv.BORDER_REFLECT: reflejar los valores para crear el borde: gfedcba | abcdefg | gfedcba 
        cv.BORDER_REFLECT_101: reflejar los valores con una ligera modificacion: gfedcb | abcdefgh | gfedcba
        cv.BORDER_REPLICATE: replicar el ultimo elemento de la imagen: aaaaa | abcdefgh | hhhhh
        cv.BORDER_WRAP: crear el borde envolviendo la imagen: cdefgh | abcdefgh | abcdefg
Retorna:
    imagen (np array) con la concatenacion de las imagenes de la piramide
"""
def piramide_laplaciana_OpenCV(imagen, niveles, padding=cv.BORDER_REFLECT):
    return unir_imagenes( lista_piramida_laplaciana_opencv(imagen, niveles, padding) )


"""
Reconstruir la imagen original que genero la piramide laplaciana a partir de esta ultima.
Se parte de la ultima imagen (del ultimo nivel) y se realiza el siguiente procedimiento:
    i) Aumentar la imagen. El metodo de upsampling utilizado es interpolacion lineal 
            (mediante la implementacion de OpenCV).
    ii) Sumar la imagen a la del nivel anterior. Esto da la imagen original de ese nivel.
    iii) Utilizando la imagen generada, volver al paso i)
Al llegar al nivel 0 se obtiene la imagen original.
Este proceso reconstruye la piramide gaussiana tambien.
Parametros:
    lista_laplaciana, lista con las imagenes de la piramide laplaciana
Retorna:
    im_reconst, np array con la imagen reconstruida
    piramide_gauss_recons, lista con las imagenes de la piramide gaussiana reconstruida
"""
def reconstruir_imagen(lista_laplaciana):
    # Obtener la ultima imagen de la piramide laplaciana
    im_reconst = lista_laplaciana[-1]
    
    # Piramide gaussiana reconstruida
    piramide_gauss_recons = [im_reconst.copy()]
    
    # Recorremos todos los niveles de la piramide desde el penultimo hasta el primero
    for i in range(-2, -len(lista_laplaciana)-1, -1):
        # Obtener el tamaño de la imagen del nivel i de la piramide laplaciana
        dim = lista_laplaciana[i].shape
        
        # Redimensionar la anterior imagen reconstruida
        im_resized = cv.resize(im_reconst, (dim[1], dim[0]), interpolation=cv.INTER_LINEAR)
        
        # La imagen reconstruida del nivel i es la suma de la imagen redimensionada y la imagen
        # del nivel i en la piramide laplaciana
        im_reconst = lista_laplaciana[i] + im_resized
        
        # Añadirla al comienzo de la piramide gaussiana
        piramide_gauss_recons.insert(0, im_reconst.copy())
    
    return im_reconst, piramide_gauss_recons


"""
Obtener la desviacion tipica del kernel de alisamiento a aplicar para obtener
la escala de un nivel dada una octava de niveles_totales niveles y partiendo
de una desviacion tipica inicial sigma_ini.
Parametros:
    num_escala, nivel de la escala dentro de la octava
    num_escalas_totales, numero total de niveles de la octava
    sigma_ini, desviacion tipica inicial de la primera escala de la octava
Retorna:
    desviacion tipica a aplicar para obtener la escala
"""
def get_sigma_octava (num_escala, num_escalas_totales, sigma_ini):
    return sigma_ini * math.sqrt(2**(2 * num_escala / num_escalas_totales) -
                                 2**(2 * (num_escala - 1) / num_escalas_totales))
    


"""
Obtener todas las escalas de una octava. Si el numero de escala totales es n,
se generan n+2 escalas para poder utilizarlas para calcular las DoG posteriormente.
Parametros:
    escala_ini, imagen inicial (escala 1)
    num_escalas_totales, numero total de escalas de la octava
    sigma_ini, desviacion tipica de la escala 1
Retorna:
    lista_escalas, lista de todas las imagenes (escalas) de la octava
"""
def obtener_escalas_gaussianas (escala_ini, num_escalas_totales, sigma_ini):
    # Añadir la escala inicial a la lista
    lista_escalas = [escala_ini.copy()]
    
    for i in range(0, num_escalas_totales+2):
        # Obtener el sigma del alisamiento
        nuevo_sigma = get_sigma_octava(i+1, num_escalas_totales, sigma_ini)
        
        mask_gauss = kernel_gaussiano(sigma=nuevo_sigma)
        
        im_padding = aplicar_padding(lista_escalas[i], cv.BORDER_REFLECT, width_top=len(mask_gauss)//2)
        
        # Alisar la imagen
        nueva_escala = convolve2D(im_padding, mask_gauss)        
        
        # Añadirla a la lista
        lista_escalas.append(nueva_escala)
    
    return lista_escalas


"""
Obtener todas las octavas.
Parametros:
    imagen_ini, imagen original de la que se parte
    num_octavas_totales, numero de octavas a obtener
    num_escalas_totales, numero de escalas de cada octava
    sigma_original, desviacion tipica del alisamiento de la imagen original
Retorna:
    lista_octavas, todas las octavas con sus respectivas escalas
"""
def obtener_octavas_gaussianas (imagen_ini, num_octavas_totales, num_escalas_totales, sigma_original):
    # Doble del tamaño de la imagen inicial
    dim = list(imagen_ini.shape)
    dim = tuple([e * 2 for e in dim])
        
    # Aumentar (upsample) la imagen inicial para obtener la primera escala de la primera octava
    escala_ini = cv.resize(imagen_ini, (dim[1], dim[0]), interpolation=cv.INTER_LINEAR)
    
    # Obtener las escalas de la octava 0
    lista_octavas = [obtener_escalas_gaussianas(escala_ini, num_escalas_totales, sigma_original)]
    
    # La primera octava ya ha sido añadida
    for i in range(0, num_octavas_totales-1):
        # Submuestrear la ultima escala verdadera de la octava anterior
        escala_ini = submuestrear_imagen(lista_octavas[i][num_escalas_totales])
        
        # Obtener las escalas de la octava i+1
        lista_octavas.append(obtener_escalas_gaussianas(escala_ini, num_escalas_totales, sigma_original))
    
    return lista_octavas


"""
Obtener las escalas DoG a partir de una octava gaussiana de escalas.
Parametros:
    lista_escalas, octava gaussiana de la que se parte
Retorna:
    lista_DoG, octava DoG
"""
def obtener_escalas_DoG (lista_escalas):
    lista_DoG = []
    
    for i in range(1, len(lista_escalas)):
        lista_DoG.append(lista_escalas[i] - lista_escalas[i-1])
    
    return lista_DoG


"""
Obtener las octavas DoG a partir de las octavas DoG de escalas.
Parametros:
    lista_octavas, octavas gaussianas de las que se parte
Retorna:
    lista_octavas_lap, octavas DoG
"""
def octavas_DoG (lista_octavas):
    lista_octavas_DoG = []
    
    for i in range(0, len(lista_octavas)):
        lista_octavas_DoG.append(obtener_escalas_DoG(lista_octavas[i]))
    
    return lista_octavas_DoG


"""
Obtener todas las octavas DoG.
Parametros:
    imagen_ini, imagen original de la que se parte
    num_octavas_totales, numero de octavas gaussianas a obtener
    num_escalas_totales, numero de escalas de cada octava gaussiana
    sigma_original, desviacion tipica del alisamiento de la imagen original
Retorna:
    lista_octavas, todas las octavas con sus respectivas escalas
"""
def obtener_octavas_DoG (imagen_ini, num_octavas_totales, num_escalas_totales, sigma_original):
    lista_octavas = obtener_octavas_gaussianas(imagen_ini, num_octavas_totales,
                                               num_escalas_totales, sigma_original)
    
    return octavas_DoG(lista_octavas)


"""
Mostrar las escalas verdaderas de cada octava gaussiana.
Parametros:
    lista_octavas, todas las octavas con sus respectivas escalas
    num_escalas_totales, numero de escalas de cada octava
"""
def mostrar_escalas_gaussianas (lista_octavas, num_escalas_totales):
    lista_titulos = []
    for i in range(0, len(lista_octavas)):
        for j in range(1, num_escalas_totales+1):
            lista_titulos.append("Octava " + str(i) + " Escala " + str(j))
    
    flat_list_octavas = [item for sublist in lista_octavas for item in sublist[1:(num_escalas_totales+1)]]
    
    mostrar_im_en_bloque(flat_list_octavas, lista_titulos, nrows=len(lista_octavas)+1, tam_fig=(6, 12))


"""
Mostrar las escalas de cada octava DoG.
Parametros:
    lista_octavas, todas las octavas con sus respectivas escalas
"""
def mostrar_escalas_DoG (lista_octavas):
    lista_titulos = []
    for i in range(0, len(lista_octavas)):
        for j in range(1, len(lista_octavas[i])+1):
            lista_titulos.append("Octava " + str(i) + " Escala " + str(j))
    
    flat_list_octavas = [item for sublist in lista_octavas for item in sublist]
    
    mostrar_im_en_bloque(flat_list_octavas, lista_titulos, nrows=len(lista_octavas), tam_fig=(8, 8))


"""
Obtener el sigma acumulado en una escala y octava determinada.
Parametros:
    sigma_original, desviacion tipica de la imagen original
    num_escalas_totales, numero de escalas reales dentro de una octava
    octava, numero de la octava
    escala, numero de la escala
Retorna:
    sigma acumulado
"""
def get_sigmak(sigma_original, num_escalas_totales, octava, escala):
    k = -num_escalas_totales + num_escalas_totales*octava + escala
    
    return sigma_original* 2**(k / num_escalas_totales)


"""
Extraer los keypoints de una imagen. Para ello se obtienen una serie de octavas (y sus
respectivas escalas) de DoG. En este espacio se buscan aquellos pixeles que sean maximo
o minimo respecto a sus 26 vecinos mas cercanos:
    los 8 pixeles que le rodean en su escala
    los 9 pixeles mas cercanos de la escala superior
    los 9 pixeles mas cercanos de la escala inferior
De entre todos los maximos encontrados se devuelven en total los num_keypoints,
por defecto 100, con mayor respuesta (mayor valor en la imagen DoG).
Parametros:
    imagen_ini, imagen original de la que se parte
    num_octavas_totales, numero de octavas gaussianas a obtener
    num_escalas_totales, numero de escalas de cada octava gaussiana
    sigma_original, desviacion tipica del alisamiento de la imagen original
    num_keypoints (opcional; por defecto 100), numero de keypoints a extraer
    diameter (opcional; por defecto 6), tamaño de los keypoints (para dibujarlos posteriormente)
Retorna:
    numpy array de objetos CV.Keypoints
"""
def extraer_keypoints (imagen_ini, num_octavas_totales, num_escalas_totales,
                       sigma_original, num_keypoints=100, radio=6):
    diameter = radio * 2
    
    lista_DoG = obtener_octavas_DoG(imagen_ini, num_octavas_totales, num_escalas_totales, sigma_original)
        
    num_escalas_DoG = len(lista_DoG[0])
    
    # Vector de posibles keypoints: se guarda la posicion del pixel, su diametro,
    # su respuesta y la octava donde se identifico
    aux = []
    dtype_raw_keypoints = [('x', float), ('y', float), ('size', float), ('response', float),
                           ('abs_response', float), ('octave', float)]
    array_raw_keypoints = np.array(aux, dtype=dtype_raw_keypoints)
    
    # Recorrer las octavas
    for num_oct in range(0, len(lista_DoG)):
        
        octava = lista_DoG[num_oct]
        
        borde_fil = octava[0].shape[0] - 1
        borde_col = octava[0].shape[1] - 1
        
        # Ejecutar para las escalas centrales
        for esc in range(1, num_escalas_DoG-1): 
            
            # Recorrer los pixeles de la escala por columnas
            for col in range(1, borde_col):
                
                # Recorrer los pixeles de la escala por filas
                for fil in range(1, borde_fil):
                    
                    # Obtener los vecinos
                    vecinos = [abs(octava[esc-1][fil][col]), abs(octava[esc+1][fil][col])] # Justo arriba y justo abajo
                                        
                    # Vecinos que lo rodean en la escala de arriba, la del pixel y la de abajo
                    for i in (-1,0,1):
                        vecinos.extend([abs(octava[esc+i][fil-1][col-1]), abs(octava[esc+i][fil-1][col]), abs(octava[esc+i][fil-1][col+1]),
                                        abs(octava[esc+i][fil][col-1])  ,                                 abs(octava[esc+i][fil][col+1]),
                                        abs(octava[esc+i][fil+1][col-1]), abs(octava[esc+i][fil+1][col]), abs(octava[esc+i][fil+1][col+1]) ])
                    
                    
                    # Añadir a la lista de posibles keypoints solo si es mayor que todos sus vecinos
                    if abs(octava[esc][fil][col]) > max(vecinos):
                        sigmak = get_sigmak(sigma_original, num_escalas_totales, num_oct, esc)
                        
                        array_raw_keypoints = np.append( array_raw_keypoints,
                                                         np.array((fil*2**(num_oct-1), col*2**(num_oct-1),
                                                                   diameter*sigmak,
                                                                   octava[esc][fil][col],
                                                                   abs(octava[esc][fil][col]),
                                                                   num_oct),
                                                                  dtype=dtype_raw_keypoints) )
        
    # Ordenar de menor a mayor respuesta absoluta
    array_raw_keypoints = np.sort(array_raw_keypoints, order='abs_response')
    
    # Guardar los num_keypoints keypoints con mayor respuesta (tanto los negativos como los positivos)
    if len(array_raw_keypoints) > num_keypoints:
        array_raw_keypoints = array_raw_keypoints[-num_keypoints:]
    
    # Construir vector de keypoints
    # Como al mostrar la imagen se muestra invertida entonces las coordenadas de los puntos han de ser invertidas
    lista_keypoints = [cv.KeyPoint(x=float(math.ceil(kp['y'])), y=float(math.ceil(kp['x'])),
                                   size=kp['size'], response=kp['response'],
                                   octave=int(kp['octave']))
                       for kp in array_raw_keypoints]
    
    return np.array(lista_keypoints)


"""
Añadir un array de keypoints a la imagen como circulos sobre las posiciones 
de los keypoints.
Paramteros:
    imagen_ini, imagen original de la que se parte
    array_keypoints, numpy array de objetos CV.Keypoints
Retorna:
    im_keypoints, imagen con los cirulos dibujados
"""
def aniadir_keypoints_im (imagen_ini, array_keypoints): 
    im_keypoints = cv.drawKeypoints(imagen_ini.astype('uint8'), array_keypoints, np.array([]),
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_keypoints


"""
Extraer los keypoints de una imagen y añadirlos como circulos sobre esta.
Parametros:
    imagen_ini, imagen original de la que se parte
    num_octavas_totales, numero de octavas gaussianas a obtener
    num_escalas_totales, numero de escalas de cada octava gaussiana
    sigma_original, desviacion tipica del alisamiento de la imagen original
    num_keypoints (opcional; por defecto 100), numero de keypoints a extraer
    diameter (opcional; por defecto 6), tamaño de los keypoints (para dibujarlos posteriormente)
Retorna:
    im_keypoints, imagen con los cirulos dibujados
"""
def extraer_aniadir_keypoints (imagen_ini, num_octavas_totales, num_escalas_totales,
                               sigma_original, num_keypoints=100, diameter=6):    
    # Extraer keypoints
    array_keypoints = extraer_keypoints (imagen_ini, num_octavas_totales, num_escalas_totales,
                                         sigma_original, num_keypoints, diameter)
    
    return aniadir_keypoints_im (imagen_ini, array_keypoints)


"""
Encontrar las correspondencias entre keypoints de dos imagenes.
Calcular las correspondencias mediante el criterio Lowe-2NN.
Parametros:
    imagen1, primera imagen original de la que se parte
    imagen2, segunda imagen original de la que se parte
Retorna:
    good_matches, las mejores correspondencias con el criterio Lowe-2NN
"""
def get_matches_Lowe2NN_openCV (imagen1, imagen2):
    # Crear detector SIFT
    sift = cv.SIFT_create()
    
    # Obtener keypoints y descriptores de ambas imagenes
    kpts1, desc1 = sift.detectAndCompute(imagen1.astype('uint8'), None)
    kpts2, desc2 = sift.detectAndCompute(imagen2.astype('uint8'), None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches.append(m)
    
    return kpts1, kpts2, good_matches


"""
Encontrar las correspondencias entre keypoints de dos imagenes.
Se calculan los keypoints y los descriptores de estos para las dos imagenes y
con ello se obtienen los matches.
Utiliza funciones de OpenCV para calcular las correspondencias mediante Brute Force.
El resultado final es una imagen que concatena las dos originales y en la que se
unen los keypoints de la primera con los de la segunda mediante las correspondencias.
Parametros:
    imagen1, primera imagen original de la que se parte
    imagen2, segunda imagen original de la que se parte
    num_matches, numero de macthes totales a extraer
    norm_type, normalizacion de BFMatcher
    cross_check
Retorna:
    im_final, imagen de correspondencias con Brute Force
"""
def matches_bruteForce_openCV (imagen1, imagen2, num_matches, 
                               norm_type=cv.NORM_L2, cross_check=True):
    # Crear detector SIFT
    sift = cv.SIFT_create()
    
    # Obtener keypoints y descriptores de ambas imagenes
    kpts1, desc1 = sift.detectAndCompute(imagen1.astype('uint8'), None)
    kpts2, desc2 = sift.detectAndCompute(imagen2.astype('uint8'), None)
    
    # Crear matcher
    bf = cv.BFMatcher(normType=norm_type, crossCheck=cross_check)

    # Encontrar correspondencias entre los descriptores
    matches = bf.match(desc1,desc2)
    
    # Mostrar num_matches correspondencias aleatoriamente
    im_final = cv.drawMatches(imagen1.astype('uint8'), kpts1,
                               imagen2.astype('uint8'), kpts2,
                               random.sample(matches, num_matches),
                               np.array([]), flags=2)
    
    return im_final


"""
Encontrar las correspondencias entre keypoints de dos imagenes.
Se calculan los keypoints y los descriptores de estos para las dos imagenes y
con ello se obtienen los matches.
Utiliza funciones de OpenCV para calcular las correspondencias mediante Lowe-2NN.
El resultado final es una imagen que concatena las dos originales y en la que se
unen los keypoints de la primera con los de la segunda mediante las correspondencias.
Parametros:
    imagen1, primera imagen original de la que se parte
    imagen2, segunda imagen original de la que se parte
    num_matches, numero de macthes totales a extraer
    norm_type, normalizacion de BFMatcher
    cross_check
Retorna:
    im_final, imagen de correspondencias con Lowe-2NN
"""
def matches_Lowe2NN_openCV (imagen1, imagen2, num_matches):
    kpts1, kpts2, matches = get_matches_Lowe2NN_openCV(imagen1, imagen2)
        
    im_final = cv.drawMatches(imagen1.astype('uint8'), kpts1,
                              imagen2.astype('uint8'), kpts2,
                              random.sample(matches, num_matches), np.array([]), flags=2)
    
    return im_final
    

"""
Obtener un canvas negro donde sea seguro que entren todas las imagenes.
Parametros:
    lista_im, lista de las imagenes que se van a usar en el mosaico
Retorna:
    numpy array de ceros con las dimensiones apropiadas
"""
def get_canvas(lista_im):
    # Obtener la suma de alturas y de anchuras de las imagenes
    len_y = 0
    len_x = 0

    for i in range(0, len(lista_im)):
        len_y = len_y + lista_im[i].shape[1]
        len_x = len_x + lista_im[i].shape[0]
            
    return np.zeros((len_x, len_y))
    

"""
Construir una imagen panoramica a partir de una lista de imagenes. La lista de imagenes
tienen solapamiento 2-a-2, es decir, la imagen i esta solapada con la imagen i+1.
El panorama se construye teniendo como imagen central la imagen del medio de la lista.
Inicialmente es un canvas negro con las dimensiones necesarias y se van añadiendo las
imagenes desde los extremos hasta el medio de la lista.
Para añadir las imagenes hay que calular una serie de homografias que transformen 
imagenes a su posicion en el panorama. Para optimizar el tiempo de calcula se va a
emplear composicion de homografias, de la siguiente forma
    · Se calcula la homografia que transforme la imagen del medio de la lista (imagen
        index_centro="indice mitad lista") al centro del canvas
    · Para cada imagen i<index_centro se calcula la homografia que transforme la 
        imagen i a la imagen i+1
    · Para cada imagen j>index_centro se calcula la homografia que transforme la
        imagen j a la imagen j-1
Tras esto se añaden las imagenes desde los extremos hasta el medio de la lista:
    · Para cada imagen (excepto la del medio) se componen las homografias que transformen
        dicha imagen sucesivamente hasta la del medio. Esta homorafia resultante se compone
        con la homografia que lleva la imagen del medio al canvas para tener la final.
        Se utiliza la funcion warpPerspective de OpenCV para pegar la imagen en el canvas.
    · Para la imagen del medio se utiliza warpPerspective con la homgrafia que lleva al canvas.
Parametros:
    lista_im, lista de imagenes del panorama
Retorna:
    canvas con la imagen panoramica
"""
def construir_panorama(lista_im):
    # Indice del medio de la lista
    index_centro = math.ceil(len(lista_im) / 2) - 1
    
    # Dimensiones de la imagen central
    x_im_centro = lista_im[index_centro].shape[0]
    y_im_centro = lista_im[index_centro].shape[1]
    
    # Obtener el canvas inicial y sus dimensiones
    canvas_fondo = get_canvas(lista_im)
    
    x_canvas = canvas_fondo.shape[0]
    y_canvas = canvas_fondo.shape[1]
    
    # Homografia que transforma la imagen del medio al canvas
    h_canvas = np.array([[1, 0, y_canvas//2 - y_im_centro//2],
                         [0, 1, x_canvas//2 - x_im_centro//2],
                         [0, 0, 1]],
                        dtype=np.float64)
    
    # Crear todas las homografias y almacenarlas en una lista
    lista_hs = [h_canvas]
    
    i_der = 0 # Para recorrer las imagenes desde el inicio hasta el medio
    i_izq = len(lista_im) - 1 # Para recorrer las imagenes desde el final hasta el medio
    while ( index_centro < i_izq ):
        # Obtener la homografia de la siguiente imagen por la izquierda con su anterior en la lista
        kpts1, kpts2, matches = get_matches_Lowe2NN_openCV (lista_im[i_izq], lista_im[i_izq-1])
    
        im_siguiente_pts = np.float32([ kpts1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        im_centro_pts = np.float32([ kpts2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
        h_relativa, mask = cv.findHomography(im_siguiente_pts, im_centro_pts, cv.RANSAC)

        # Insertarla en la respectiva posicion en la lista de homografias
        lista_hs.insert(len(lista_im)-i_izq, h_relativa)
                
        if i_der < index_centro:
            # Obtener la homografia de la siguiente imagen por la derecha con su siguiente en la lista
            kpts1, kpts2, matches = get_matches_Lowe2NN_openCV (lista_im[i_der], lista_im[i_der+1])
    
            im_siguiente_pts = np.float32([ kpts1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            im_centro_pts = np.float32([ kpts2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
            h_relativa, mask = cv.findHomography(im_siguiente_pts, im_centro_pts, cv.RANSAC)

            # Insertarla en la respectiva posicion en la lista de homografias
            lista_hs.insert(i_der, h_relativa)
        
        i_izq = i_izq - 1
        i_der = i_der + 1
        
    i_der = 0 # Para recorrer las imagenes desde el inicio hasta el medio
    i_izq = len(lista_im) - 1 # Para recorrer las imagenes desde el final hasta el medio
    while ( index_centro < i_izq ):
        # Componer todas las homografias que llevan a la siguiente imagen por la izquierda hasta el canvas
        h_total_izq = np.dot(lista_hs[i_izq-1], lista_hs[i_izq])
        
        i_aux = i_izq - 2
        while(index_centro <= i_aux):
            h_total_izq = np.dot(lista_hs[i_aux], h_total_izq)
            i_aux = i_aux - 1
        
        # Transformar la imagen al canvas con la composicion de homografias
        canvas_fondo = cv.warpPerspective(lista_im[i_izq], h_total_izq, (y_canvas,x_canvas),
                                          borderMode=cv.BORDER_TRANSPARENT, dst=canvas_fondo)
        
        if i_der < index_centro:
            # Componer todas las homografias que llevan a la siguiente imagen por la izquierda hasta el canvas
            h_total_der = np.dot(lista_hs[i_der+1], lista_hs[i_der])
        
            i_aux = i_der + 2
            while(i_aux <= index_centro):
                h_total_der = np.dot(lista_hs[i_aux], h_total_der)
                i_aux = i_aux + 1
            
            # Transformar la imagen al canvas con la composicion de homografias
            canvas_fondo = cv.warpPerspective(lista_im[i_der], h_total_der, (y_canvas,x_canvas),
                                              borderMode=cv.BORDER_TRANSPARENT, dst=canvas_fondo)
                
        i_izq = i_izq - 1
        i_der = i_der + 1
            
    # Finalmente añadir la imagen central al canvas
    canvas_fondo = cv.warpPerspective(lista_im[index_centro], h_canvas, (y_canvas,x_canvas),
                                        borderMode=cv.BORDER_TRANSPARENT, dst=canvas_fondo)
    
    return recortar_canvas(canvas_fondo)
    

"""
Eliminar las filas y columnas sin contenido en el canvas. El canvas inicial se crea con
unas dimensiones suficientes para poder albergar el panorama, por lo que al final de la
creacion de este es necesario eliminar aquellas filas y columnas totalmente vacias (negras).
Parametros:
    canvas, numpy array que contiene el canvas con la imagen panoramica
Retorna:
    canvas_final, numpy array obtenido a partir de canvas eliminando las filas y columnas
        totalmente negras
"""
def recortar_canvas(canvas):
    # Mascara con los indices de los pixeles diferentes de cero
    diferentes_cero = np.nonzero(canvas)
    
    # Primera fila y columna con contenido
    primera_columna = np.min(diferentes_cero[0])
    primera_fila = np.min(diferentes_cero[1])
    
    # Primera fila y columna sin contenido
    ultima_columna  = np.max(diferentes_cero[0]) + 1
    ultima_fila = np.max(diferentes_cero[1]) + 1
    
    # Eliminar desde la primera fila y columna sin contenido hasta el final del canvas
    canvas_final = np.delete(canvas, np.arange(ultima_columna, canvas.shape[0]), 0)
    canvas_final = np.delete(canvas_final, np.arange(ultima_fila, canvas.shape[1]), 1)
    
    # Eliminar desde el inicio del canvas hasta la primera fila y columna con contenido
    canvas_final = np.delete(canvas_final, np.arange(0, primera_columna), 0)
    canvas_final = np.delete(canvas_final, np.arange(0, primera_fila), 1)
    
    return canvas_final
    
    


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
print('*'*60, '\nEjercicio 1A, 1B y 1C. Calcular las escalas y octavas gaussianas.\n')

yosemite1_grey = leeimagen(path_yosemite1, 0)
yosemite2_grey = leeimagen(path_yosemite2, 0)


num_escalas_totales = 3
num_octavas_totales = 4
sigma_original = 0.8


lista_octavas_1 = obtener_octavas_gaussianas(yosemite1_grey, num_octavas_totales,
                                              num_escalas_totales, sigma_original)

lista_octavas_2 = obtener_octavas_gaussianas(yosemite2_grey, num_octavas_totales,
                                              num_escalas_totales, sigma_original)


mostrar_escalas_gaussianas(lista_octavas_1, num_escalas_totales)
mostrar_escalas_gaussianas(lista_octavas_2, num_escalas_totales)



input("\n--- Pulsar tecla para continuar ---\n")


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
#print('*'*60, '\nEjercicio 1D. Calcular las escalas y octavas DoG y obtener los extremos.\n')

#yosemite1_grey = leeimagen(path_yosemite1, 0)
# yosemite2_grey = leeimagen(path_yosemite2, 0)


#num_escalas_totales = 3
#num_octavas_totales = 4
#sigma_original = 1.6
#num_keypoints = 100
#radio = 6


#lista_octavas_DoG_1 = obtener_octavas_DoG(yosemite1_grey, num_octavas_totales,
#                                           num_escalas_totales, sigma_original)

# lista_octavas_DoG_2 = obtener_octavas_DoG(yosemite2_grey, num_octavas_totales,
#                                           num_escalas_totales, sigma_original)


#mostrar_escalas_DoG(lista_octavas_DoG_1)
#mostrar_escalas_DoG(lista_octavas_DoG_2)


#array_keypoints1 = extraer_keypoints (yosemite1_grey, num_octavas_totales, num_escalas_totales,
#                                      sigma_original, num_keypoints, radio)

#yosemite1_keypoints = aniadir_keypoints_im(yosemite1_grey, array_keypoints1)

#mostrar_imagen(yosemite1_keypoints, "Yosemite 1 con keypoints")


# array_keypoints2 = extraer_keypoints (yosemite2_grey, num_octavas_totales, num_escalas_totales,
#                                       sigma_original, num_keypoints, radio)

# yosemite2_keypoints = aniadir_keypoints_im(yosemite2_grey, array_keypoints2)

# mostrar_imagen(yosemite2_keypoints, "Yosemite 2 con keypoints")


# # Realizar lo mismo pero con las funciones de OpenCV
# # Crear detector SIFT
# sift = cv.SIFT_create()

# # Obtener keypoints y descriptores
# kpts, desc = sift.detectAndCompute(yosemite1_grey.astype('uint8'), None)
# im_keypoints = cv.drawKeypoints(yosemite1_grey.astype('uint8'),
#                                 np.random.choice(kpts, num_keypoints),
#                                 np.array([]),
#                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
# mostrar_imagen(im_keypoints, "Yosemite 1 con keypoints (OpenCV)")


# # Crear detector SIFT
# sift = cv.SIFT_create()

# # Obtener keypoints y descriptores
# kpts, desc = sift.detectAndCompute(yosemite2_grey.astype('uint8'), None)
# im_keypoints = cv.drawKeypoints(yosemite2_grey.astype('uint8'),
#                                 np.random.choice(kpts, num_keypoints),
#                                 np.array([]),
#                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
# mostrar_imagen(im_keypoints, "Yosemite 2 con keypoints (OpenCV)")


# input("\n--- Pulsar tecla para continuar ---\n")


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# print('*'*60, '\nEjercicio 2. Extraer matches entre dos imagenes.\n')

# yosemite1_grey = leeimagen(path_yosemite1, 0)
# yosemite2_grey = leeimagen(path_yosemite2, 0)

# num_matches = 100


#im_matches_bruteForce = matches_bruteForce_openCV(yosemite1_grey, yosemite2_grey, num_matches)

# im_matches_Lowe2NN = matches_Lowe2NN_openCV(yosemite1_grey, yosemite2_grey, num_matches)

# mostrar_imagen(im_matches_bruteForce, "Yosemite 1 y Yosemite 2 matches (Brute Force)")
# mostrar_imagen(im_matches_Lowe2NN, "Yosemite 1 y Yosemite 2 matches (Lowe-2NN)")


# input("\n--- Pulsar tecla para continuar ---\n")


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# =============================================================================
# print('*'*60, '\nEjercicio 3. Crear un mosaico de tres imagenes.\n')
# mosacio1 = leeimagen(path_mosaico_1, 1)
# mosacio2 = leeimagen(path_mosaico_2, 1)
# mosacio3 = leeimagen(path_mosaico_3, 1)
# 
# lista_mosaico = [mosacio1, mosacio2, mosacio3]
# 
# num_matches = 100
# 
# mosaico_final = construir_panorama(lista_mosaico)
# 
# mostrar_imagen(mosaico_final, "Mosaico de tres imagenes")
# 
# 
# input("\n--- Pulsar tecla para continuar ---\n")
# =============================================================================


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# =============================================================================
# print('*'*60, '\nBonus 2. Crear un mosaico.\n')
# 
# lista_mosaico = leer_imagenes_directorio(path_mosaico, 1)
# 
# num_matches = 100
# 
# mosaico_final = construir_panorama(lista_mosaico)
# 
# mostrar_imagen(mosaico_final, "Mosaico")
# =============================================================================


