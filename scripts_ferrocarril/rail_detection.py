
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

        6-)Carga de datos a CloudCompare

        7-)Reconstrucción de railes y eje de via (...en proceso de desarrollo)

Resultado:
        -Se consigue filtrar los puntos que pertencen a las vias, pero sigue existiendo ruido
        -Queda pendiente reconstruir las lineas de railes y obtener ejes de via
"""

# Manejo de rutas de ejecución
import os
import sys


parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#modulos de cloudcompare
import pycc
import cccorelib

#otros
import time

#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, makePyccPointCloudObject
from utils.utilities import getCloudAsArray, getGeometricalInfo, getTrackPoints


#Aqui inicia la ejecución del código
if __name__=="__main__":
    print("\n","-"*45)
    print('Analizando nube de puntos...')
    print("-"*45)


    global_start = time.time()

#1-)
    #============================================================================================
    #===========================carga de los datos desde CloudCompare============================
    #============================================================================================
    cc, point_cloud , xyz = getCloud()
    point_cloud_name = point_cloud.getName()


#2-)
    #============================================================================================
    #==========Remuestreo de la nube de puntos a 1 punto cada 10cm en las 3 dimensiones==========
    #============================================================================================
    print("\n","-"*45)
    print('Remuestreo de puntos...')
    print("-"*45)
    start = time.time()

    distance = 0.1
    ref_subsample = cccorelib.CloudSamplingTools.resampleCloudSpatially(cloud=point_cloud,
                                                                        minDistance= distance,
                                                                        modParams=cccorelib.CloudSamplingTools.SFModulationParams(False),
                                                                        progressCb=pycc.ccProgressDialog().start())
    spatial_subsample_cloud = point_cloud.partialClone(ref_subsample)
    #spatial_subsample_cloud.setName('spatially subsample 10cm')
    #cc.addToDB(spatial_subsample_cloud)
    del point_cloud

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el remuestreo de puntos")



#3-)
    #============================================================================================
    #=======Conversión de la nube de puntos a ndarray de numpy (x,y,z,globalScalarFields)========
    #============================================================================================
    cloudAsArray, globalScalarFieldsNames = getCloudAsArray(spatial_subsample_cloud)
    del spatial_subsample_cloud


#4-)
    #============================================================================================
    #========Cálculo de características geométricas basadas en el vecindario de cada punto=======
    #============================================================================================
    print("\n","-"*45)
    print('Cálculo de caracteristicas geométricas...')
    print("-"*45)
    start = time.time()

    n=15 #15 vecinos más cercanos
    cloud_geometricalInfo, globalScalarFieldsNames = getGeometricalInfo(cloud=cloudAsArray, globalScalarfields=globalScalarFieldsNames, n=n)
    del cloudAsArray

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el calculando caracteristicas")



#5-)
    #============================================================================================
    #==============================Obtención inicial de track points=============================
    #============================================================================================
    print("\n","-"*45)
    print('Identificando track points...')
    print("-"*45)
    start = time.time()

    #segmentación de la nube de puntos
    _, cell_clouds = getCellPointCloud(cloud_geometricalInfo, cell_size=1)
    del cloud_geometricalInfo

    #obtención de track_points
    track_point_cloud, globalScalarFieldsNames = getTrackPoints(clouds=cell_clouds, scalarFields=globalScalarFieldsNames)
    del cell_clouds

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en la identificación de track points")



#6-)
    #============================================================================================
    #===============================Carga de datos a CloudCompare================================
    #============================================================================================
    print("\n","-"*45)
    print('Fin del proceso!')
    print("-"*45)

    #conversión de array a pycc.PoinCloud
    pc = makePyccPointCloudObject(track_point_cloud, globalScalarFieldsNames, f"{point_cloud_name}_initial_track_point")
    pc.setCurrentDisplayedScalarField(pc.getScalarFieldIndexByName("RANSAC_Direction"))
    cc.addToDB(pc)

    global_end = time.time()
    elapsed_time = (global_end- global_start)/60
    print("        ", f"{elapsed_time:.2f} min: en todo el proceso de análisis")

    #actualizar la GUI de CloudCompare
    cc.updateUI()



#7-)en proceso de desarrollo...
    #============================================================================================
    #==========================Reconstrucción de railes y eje de via ============================
    #============================================================================================




