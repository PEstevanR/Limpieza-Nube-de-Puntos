
#==============================================================================================================
#=============================================Script en desarrollo=============================================
#==============================================================================================================
"""
Objetivo:
        Obtener los puntos que hacen parte de cada carril de via como paso previo a la obtención del eje de via
Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare

        2-)Remuestreo de la nube de puntos a 1 punto cada 10cm en las 3 dimensiones

        3-)Conversión de la nube de puntos a ndarray de numpy de la forma (x,y,z,globalScalarFields)

        4-)Cálculo de características geométricas basadas en el vecindario de cada punto (15 vecinos más cercanos)

        5-)Filtrado inicial de trackpoints acorde a varias condiciones en celdas de tamaño 'cell_size=1'

        6-)Ajuste local con RANSAC para obtener inliers de cada modelo ajustado por celda

        7-)Carga de datos a CloudCompare

        8-)Reconstrucción de railes y eje de via (...en proceso de desarrollo)

Resultado:
        -Se consigue filtrar los puntos que pertencen a las vias, pero sigue existiendo ruido
        -

"""

# Manejo de rutas de ejecución
import os
import sys

current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#modulos de cloudcompare
import pycc
import cccorelib

#otros
import time
import numpy as np

#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, makePyccPointCloudObject, mergePointClouds
from utils.utilities import getCloudAsArray, getGeometricalInfo, getTrackPoints, customRansac


#Aqui inicia la ejecución del código
if __name__=="__main__":

    #para medir el tiempo de ejecución total
    global_start = time.time()

#1-)
    #============================================================================================
    #===========================carga de los datos desde CloudCompare============================
    #============================================================================================
    cc, point_cloud , _ = getCloud()


#2-)
    #============================================================================================
    #==========Remuestreo de la nube de puntos a 1 punto cada 10cm en las 3 dimensiones==========
    #============================================================================================
    distance = 0.1
    ref_subsample = cccorelib.CloudSamplingTools.resampleCloudSpatially(cloud=point_cloud,
                                                                        minDistance= distance,
                                                                        modParams=cccorelib.CloudSamplingTools.SFModulationParams(False),
                                                                        progressCb=pycc.ccProgressDialog().start())
    spatial_subsample_cloud = point_cloud.partialClone(ref_subsample)
    #spatial_subsample_cloud.setName('spatially subsample 10cm')
    #cc.addToDB(spatial_subsample_cloud)
    del point_cloud


#3-)
    #============================================================================================
    #=======Conversión de la nube de puntos a ndarray de numpy (x,y,z,globalScalarFields)========
    #============================================================================================
    start = time.time()
    cloudAsArray, globalScalarFieldsNames = getCloudAsArray(spatial_subsample_cloud)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido para conversión de nube de puntos a ndarray de numpy: {elapsed_time:.2f} min')
    del spatial_subsample_cloud


#4-)
    #============================================================================================
    #========Cálculo de características geométricas basadas en el vecindario de cada punto=======
    #============================================================================================
    start = time.time()
    n=15 #15 vecinos más cercanos
    cloud_geometricalInfo, globalScalarFieldsNames = getGeometricalInfo(cloud=cloudAsArray, globalScalarfields=globalScalarFieldsNames, n=n)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido para calcular características geométricas: {elapsed_time:.2f} min')
    del cloudAsArray


#5-)
    #============================================================================================
    #==============================Obtención inicial de track points=============================
    #============================================================================================
    start = time.time()

    #segmentación de la nube de puntos
    _, clouds = getCellPointCloud(cloud_geometricalInfo, cell_size=1)

    #obtención de track_points
    track_clouds = getTrackPoints(clouds)

    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido en el proceso de obtención inicial de track_points: {elapsed_time:.2f} min')
    del cloud_geometricalInfo, clouds


#6-)
    #============================================================================================
    #===================================Ajuste local de RANSAC===================================
    #============================================================================================

    #array numpy vacio con el mismo número de columnas que los arrays en track_clouds
    f = track_clouds[0].shape[1]
    track_point_cloud = np.empty((0,f+1)) #+1 porque dentro del bucle for, se añade una columna más al array (direction)

    #iteramos los segmentos de la nube de puntos para hacer un ajuste RANSAC por celdas y obtener los inliers del modelo ajustado
    for track_points in track_clouds:
        #Ajuste RANSAC
        inliers, outliers, m = customRansac(track_points)

    ##Condición 4
        #Si los puntos inliers del modelo RANSAC son más del 90% nos quedamos con ellos
        condition4 = (inliers.shape[0] / track_points.shape[0])*100

        if condition4 > 90.0:
            #filtrar en cada celda los inlier con la mascara de inliers del modelo RANSAC
            track_points = track_points[inliers]

            #añadiendo la dirección de la recta de regresión a la nube de puntos como un atributo más
            direction = np.full(shape=(track_points.shape[0]), fill_value=m)
            track_points = np.append(track_points, np.expand_dims(direction, axis=1), 1)

            #guardando el resultado
            track_point_cloud = np.vstack((track_point_cloud, track_points))


#7-)
    #============================================================================================
    #===============================Carga de datos a CloudCompare================================
    #============================================================================================
    #eliminando puntos duplicados
    track_points = np.unique(track_point_cloud, axis=0)

    #actualizando los nombres de los campos escalares
    globalScalarFieldsNames.append("RANSAC_Direction")

    #conversión de array a pycc.PoinCloud
    pc = makePyccPointCloudObject(track_point_cloud, globalScalarFieldsNames, "track_points")
    pc.setCurrentDisplayedScalarField(pc.getScalarFieldIndexByName("RANSAC_Direction"))
    cc.addToDB(pc)

    global_end = time.time()
    elapsed_time = (global_end- global_start)/60
    print(f'Tiempo transcurrido durante todo el proceso: {elapsed_time:.2f} min')

    #actualizar la GUI de CloudCompare
    cc.updateUI()


    #====================================================
    #Para visualizar las celdas en CloudCompare
    #====================================================
    #grid = pycc.ccPointCloud()
    #for cell in cells:
    #    for point in cell:
    #        grid.addPoint(cccorelib.CCVector3(point[0], point[1], 0))
    #grid.setName("Grid")
    #cc.addToDB(grid)
    #====================================================
    #====================================================

#8-)reconstrucción de railes y eje de via (...en proceso de desarrollo)

    print("PROCESO FINALIZADO")



