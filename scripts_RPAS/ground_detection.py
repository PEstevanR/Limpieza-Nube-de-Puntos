#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================
"""
Objetivo:
        Generar una superficie de comparación para filtrar puntos de terreno (ground points) en nubes de puntos
        obtenidas mediante técnicas de fotogrametría aérea con imágenes captadas con RPAS

Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare

        2-)Remuestreo de la nube de puntos a 1 punto cada 10cm en las 3 dimensiones

        3-)Reducción de puntos al 50% de la nube remuestreada

        4-)Conversión de la nube de puntos a array numpy de la forma (x,y,z,globalScalarFields)

        5-)Filtro geométrico de puntos terreno con el algoritmo Cloth Simulation Filter (CSF) en celdas de 50m

        6-)Cálculo de índices espectrales para cada punto con los campos escalares RGB

        7-)Cálculo de características geométricas basadas en el vecindario de cada punto (15 vecinos más cercanos)

        8-)Clasificación no supervisada con base en características espectrales y características geométricas (algoritmo GMM)

        9-)Proceso de creación de malla de comparación

        10-)Carga de datos a CloudCompare

Resultado:
        -Se consigue filtrar puntos de terreno y una primera aproximación a la superficie de comparación.
        -La herramienta funciona adecuadamente en zonas llanas o semionduladas
        -
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
import numpy as np
from scipy.spatial import Delaunay


#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, makePyccPointCloudObject
from utils.utilities import getCloudAsArray, filterCSF, getSpectralInfo, getGeometricalInfo, clusteringGMM


#Aqui inicia la ejecución del código
if __name__ == "__main__":

    print("\n","-"*45)
    print('Analizando nube de puntos...')
    print("-"*45)
    global_start = time.time()



#1-)
    #============================================================================================
    #===========================carga de los datos desde CloudCompare============================
    #============================================================================================
    cc, point_cloud, _ = getCloud()
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

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el remuestreo de puntos")



#4-)
    #============================================================================================
    #========Conversión de la nube de puntos a array de numpy (x,y,z,globalScalarFields)=========
    #============================================================================================
    cloudAsArray, globalScalarFieldsNames = getCloudAsArray(random_subsample_cloud)
    del random_subsample_cloud



#5-)
    #============================================================================================
    #===Filtrado geométrico de puntos terreno -Cloth Simulation Filter (CSF)- en celdas de 50m===
    #============================================================================================
    print("\n","-"*45)
    print('Filtrado geométrico CSF...')
    print("-"*45)
    start = time.time()

    cell_size = 50
    _, cellPointCloud = getCellPointCloud(cloudAsArray, cell_size)
    del cloudAsArray

    #parámetros del algoritmo CSF
    smooth = True     # manejar pendientes pronunciadas después de la simulación
    threshold = 0.5   # umbral de clasificación puntos terreno y no terreno
    resolution = 0.5  # resolución de la tela
    rigidness = 2     # 1)montaña y vegetación densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    interations = 500 # número de iteraciones

    params = {'bSloopSmooth':smooth, 'class_threshold':threshold, 'cloth_resolution':resolution, 'rigidness':rigidness, 'interations':interations}
    ground_csf, _ = filterCSF(cellPointCloud, params)
    del cellPointCloud

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el filtrado geométrico")



#6-)
    #============================================================================================
    #===================Cálculo de índices espectrales con campos escalares RGB==================
    #============================================================================================
    print("\n","-"*45)
    print('Cálculo de caracteristicas espectrales...')
    print("-"*45)
    start = time.time()

    cloud_espectralInfo, globalScalarFieldsNames = getSpectralInfo(cloud=ground_csf, scalarfields=globalScalarFieldsNames)
    del ground_csf

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el calculando de caracteristicas")



#7-)
    #============================================================================================
    #========Cálculo de características geométricas basadas en el vecindario de cada punto=======
    #============================================================================================
    print("\n","-"*45)
    print('Cálculo de caracteristicas geométricas...')
    print("-"*45)
    start = time.time()

    n=15 #15 vecinos más cercanos
    cloud_geometricalInfo, globalScalarFieldsNames = getGeometricalInfo(cloud=cloud_espectralInfo, globalScalarfields=globalScalarFieldsNames, n=n)
    del cloud_espectralInfo

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en el calculando caracteristicas")




#8-)
    #============================================================================================
    #==============Clasificación no supervisada (algoritmo Gaussian Mixture Model)===============
    #============================================================================================
    print("\n","-"*45)
    print('Agrupando puntos...')
    print("-"*45)
    start = time.time()

    cell_size = 15 #ventana de análisis
    cloud_ground_gmm, globalScalarFieldsNames = clusteringGMM(cloud_geometricalInfo, globalScalarFieldsNames, cell_size)
    del cloud_geometricalInfo

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en la agrupando de puntos")




#9-)
    #============================================================================================
    #========================Proceso de creación de malla de comparación=========================
    #============================================================================================
    print("\n","-"*45)
    print('Triangulación de superficie...')
    print("-"*45)
    start = time.time()

    #triangulación de Delaunay
    faces = Delaunay(cloud_ground_gmm[:,:2]).simplices

    # creación de malla
    pc = makePyccPointCloudObject(cloud_ground_gmm, globalScalarFieldsNames, f"{point_cloud_name}_ground_points")
    mesh = pycc.ccMesh(pc)
    for (i1, i2, i3) in faces:
        mesh.addTriangle(i1, i2, i3)
    mesh.setName(f"{point_cloud_name}_mesh_ground_points")

    stop = time.time()
    elapsed_time = (stop-start)/60
    print("        ", f"{elapsed_time:.2f} min: en la triangulación")


#10-)
    #============================================================================================
    #===============================Carga de datos a CloudCompare================================
    #============================================================================================
    print("\n","-"*45)
    print('Fin del proceso!')
    print("-"*45)

    cc.addToDB(pc)
    cc.addToDB(mesh)

    global_end = time.time()
    elapsed_time = (global_end- global_start)/60
    print("        ", f"{elapsed_time:.2f} min: en todo el proceso de análisis")

    #actualizar la GUI de CloudCompare
    cc.updateUI()



