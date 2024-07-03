#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================
"""
Objetivo:
        Generar una superficie de comparación para filtrar puntos de terreno (ground points) en nubes de puntos
        obtenidas mediante técnicas de fotogrametría aérea con imágenes capturadas con RPAS

Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare

        2-)Remuestreo de la nube de puntos a 1 punto cada 10cm en las 3 dimensiones

        3-)Reducción de puntos al 50% de la nube remuestreada

        4-)Conversión de la nube de puntos a ndarray de numpy de la forma (x,y,z,globalScalarFields)

        5-)Filtro (GEOMETRICO) de puntos terreno con el algoritmo Cloth Simulation Filter (CSF) en celdas de 50m

        6-)Cálculo de índices espectrales para cada punto con los campos escalares RGB

        7-)Cálculo de características geométricas basadas en el vecindario de cada punto (15 vecinos más cercanos)

        8-)Clasificación no supervisada con base en características espectrales y características geométricas (algoritmo GMM)

        9-)Proceso de creación de malla de comparación

        10-)Carga de datos a CloudCompare

Resultado:
"""

# Manejo de rutas de ejecución
import os
import sys

current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
#parent_dir="D:\LimpiezaNubes"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#modulos de cloudcompare
import pycc
import cccorelib

#otros
import time
import numpy as np
import dendromatics as dm
from scipy.spatial import Delaunay


#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, mergePointClouds, makePyccPointCloudObject
from utils.utilities import filterCSF, getCloudAsArray, getSpectralInfo, getGeometricalInfo, GMM_clustering


#Aqui inicia la ejecución del código
if __name__ == "__main__":

    global_start = time.time()
    print('Inicio del proceso de obtención de puntos terreno')


#1-)
    #============================================================================================
    #===========================carga de los datos desde CloudCompare============================
    #============================================================================================
    cc, point_cloud, _ = getCloud()


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
    #=====================Reducción de puntos al 50% de la nube remuestreada=====================
    #============================================================================================
    percent = 0.5
    reduccion = int(np.ceil(spatial_subsample_cloud.size()*percent))
    ref_subsample = cccorelib.CloudSamplingTools.subsampleCloudRandomly(cloud=spatial_subsample_cloud,
                                                                        newNumberOfPoints=reduccion,
                                                                        progressCb=pycc.ccProgressDialog().start())
    random_subsample_cloud = spatial_subsample_cloud.partialClone(ref_subsample)
    #random_subsample_cloud.setName('random subsample 50%')
    #cc.addToDB(random_subsample_cloud)
    del spatial_subsample_cloud



#4-)
    #============================================================================================
    #=======Conversión de la nube de puntos a ndarray de numpy (x,y,z,globalScalarFields)========
    #============================================================================================
    start = time.time()
    cloudAsArray, globalScalarFieldsNames = getCloudAsArray(random_subsample_cloud)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido para conversión de nube de puntos a ndarray de numpy: {elapsed_time:.2f} min')
    del random_subsample_cloud



#5-)
    #============================================================================================
    #===Filtrado geométrico de puntos terreno -Cloth Simulation Filter (CSF)- en celdas de 50m===
    #============================================================================================
    start = time.time()
    cell_size = 50
    _, cellPointCloud = getCellPointCloud(cloudAsArray, cell_size)
    del cloudAsArray

    #parametros del algoritmo CSF
    smooth = True     # manejar pendientes pronunciadas después de la simulación
    threshold = 0.5   # umbral de clasificación puntos terreno y no terreno
    resolution = 0.5  # resolución de la tela
    rigidness = 2     # 1)montaña y vegetación densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    interations = 500 # número de iteraciones
    params = [smooth, threshold, resolution, rigidness, interations]
    ground_csf, _ = filterCSF(cellPointCloud, params)

    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido en filtrado geométrico (CSF): {elapsed_time:.2f} min')
    del cellPointCloud



#6-)
    #============================================================================================
    #===================Cálculo de índices espectrales con campos escalares RGB==================
    #============================================================================================
    start = time.time()
    cloud_espectralInfo, globalScalarFieldsNames = getSpectralInfo(cloud=ground_csf, scalarfields=globalScalarFieldsNames)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido para calcular características espectrales: {elapsed_time:.2f} min')
    del ground_csf



#7-)
    #============================================================================================
    #========Cálculo de características geométricas basadas en el vecindario de cada punto=======
    #============================================================================================
    start = time.time()
    n=15 #15 vecinos más cercanos
    cloud_geometricalInfo, globalScalarFieldsNames = getGeometricalInfo(cloud=cloud_espectralInfo, globalScalarfields=globalScalarFieldsNames, n=n)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido para calcular características geométricas: {elapsed_time:.2f} min')
    del cloud_espectralInfo



#8-)
    #============================================================================================
    #==============Clasificación no supervisada (algoritmo Gaussian Mixture Model)===============
    #============================================================================================
    start = time.time()
    cell_size = 15 #ventana de análisis
    cloud_ground_gmm, globalScalarFieldsNames = GMM_clustering(cloud_geometricalInfo, globalScalarFieldsNames, cell_size)
    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido en el proceso de clustering: {elapsed_time:.2f} min')
    del cloud_geometricalInfo



#9-)
    #============================================================================================
    #========================Proceso de creación de malla de comparación=========================
    #============================================================================================
    start = time.time()

    #triangulación de Delaunay e índices de los triángulos
    indices = Delaunay(cloud_ground_gmm[:,:2]).simplices

    # creación de malla
    pc = makePyccPointCloudObject(cloud_ground_gmm, globalScalarFieldsNames, "ground_points")
    mesh = pycc.ccMesh(pc)
    for (i1, i2, i3) in indices:
        mesh.addTriangle(i1, i2, i3)
    mesh.setName("mesh_ground_points")

    stop = time.time()
    elapsed_time = (stop-start)/60
    print(f'Tiempo transcurrido en el proceso de generación de malla: {elapsed_time:.2f} min')



#10-)
    #============================================================================================
    #===============================Carga de datos a CloudCompare================================
    #============================================================================================
    ground_cloud= mergePointClouds([pc], name_merge='ground_points')
    ground_cloud.setCurrentDisplayedScalarField(ground_cloud.getScalarFieldIndexByName("Zcoord"))

    cc.addToDB(ground_cloud)
    cc.addToDB(mesh)

    global_end = time.time()
    elapsed_time = (global_end- global_start)/60
    print(f'Tiempo transcurrido durante todo el proceso: {elapsed_time:.2f} min')

    #actualizar la GUI de CloudCompare
    cc.updateUI()



