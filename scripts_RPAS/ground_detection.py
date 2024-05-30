#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================
"""
Objetivo:
        filtrar los puntos que son de terreno (ground points) para generar curvas de nivel

Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare

        2-)Reducción de puntos al 50% de forma aleatoria, para hacer más agil el procesamiento de datos

        3-)Reducción de ruido con el filtro 'filter_noise' de cloudcompare

        4-)Obtención de información espectral a partir de los campos escalares RGB

        5-)Segmentación de la nube de puntos por celdas de paso 50m

        6-)Filtrado de puntos de terreno natural con el algoritmo Cloth Simulation Filter (filtrado geométrico)

        7-)Filtrado espectral empleando los indices ExG y el negativo del campo escalar rojo

        8-)Interpolación de curvas de nivel

        9-)Carga de nubes de puntos a CloudCompare

Resultado:
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
import numpy as np

#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, mergePointClouds, makePyccPointCloudObject
from utils.utilities import filterCSF, espectralInfo






#Aqui inicia la ejecución del código
if __name__=="__main__":

    print("Inicio de ejecución")

#1-)cargamos los datos desde CloudCompare
    cc, point_cloud, _ = getCloud()

#2-)reducción de puntos al 50%
    reduccion = point_cloud.size()//2
    subsample = cccorelib.CloudSamplingTools.subsampleCloudRandomly(cloud=point_cloud, newNumberOfPoints=reduccion)
    random_subsample_cloud = point_cloud.partialClone(subsample)
    #random_subsample.setName('subsample 50%')
    #cc.addToDB(random_subsample)


#3-)Eliminación de ruido
    #refcloud_2 = cccorelib.CloudSamplingTools.sorFilter(cloud=random_subsample, knn=6, nSigma=1.0)
    noisefilter = cccorelib.CloudSamplingTools.noiseFilter(cloud=random_subsample_cloud,
                                                          kernelRadius=0.1,
                                                          nSigma=1.0,
                                                          removeIsolatedPoints = True,
                                                          useKnn = False,
                                                          useAbsoluteError=False,
                                                          )
    denoised_cloud = random_subsample_cloud.partialClone(noisefilter)
    #denoised_cloud.setName('filtro de ruido')
    #cc.addToDB(denoised_cloud)


#4-)Extrayendo información extra de la nube de puntos y conversion a ndarray numpy
    xyz, scalar_fields = espectralInfo(denoised_cloud)

#5-)segmentación de la nube de puntos por celdas
    cell_size = 50
    cells, clouds = getCellPointCloud(xyz, cell_size)

    print("segmentacion hecha")

#6-)filtro (GEOMETRICO) de puntos terreno con Cloth Simulation Filter (CSF)
    #parametros del algoritmo
    smooth = True    # manejar pendientes pronunciadas después de la simulación
    threshold = 0.2  # umbral de clasificación puntos terreno y no terreno
    resolution = 1 # resolución de la tela
    rigidness = 2    # 1)montaña y vegetacion densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    interations = 500# numero de iteraciones
    params = [smooth, threshold, resolution, rigidness, interations]
    ground_csf, n_ground_csf = filterCSF(clouds, params)

    print("filtrado geométrico hecho")

#7-)filtro (ESPECTRAL) basado en los indices Negative_R y ExG
    #umbrales
    umbral_negative_R = 153
    umbral_ExG = 25
    #recuperamos la posición de cada indice dentro del array de la nube de puntos
    if 'Negative_R' in scalar_fields and 'ExG' in scalar_fields:
        idx1 = list(scalar_fields).index('ExG')+3
        idx2 = list(scalar_fields).index('Negative_R')+3

        ground_espectral = []
        for cloud in ground_csf:
            #filtro 1: todos los puntos con un valor en el indice ExG menor o igual que 25 son de terreno
            #filtro 2: todos los puntos con un valor en el indice Negative_R menor o igual que 255*0.6 = 153 son de terreno
            mask_ground = np.logical_or(cloud[:,idx1] <= umbral_ExG, cloud[:,idx2] <= umbral_negative_R)
            mask_n_ground = np.logical_not(mask_ground)

            #Extraemos los puntos filtrados y los almacenamos en una nueva lista
            ground_points = cloud[mask_ground]
            ground_espectral.append(ground_points)

            #Extraemos los puntos que no se han filtrado y los añadimos a la lista n_ground_csf (elementos sobre el terreno)
            n_ground_points = cloud[mask_n_ground]
            n_ground_csf.append(n_ground_points)


    print("filtrado espectral hecho")
#8-) obtención de curvas de nivel


#9-)Carga de nubes de puntos a CloudCompare
    cc_ground_clouds = [makePyccPointCloudObject(cloud, scalar_fields, "Ground points") for cloud in ground_espectral]
    cc_n_ground_clouds = [makePyccPointCloudObject(cloud, scalar_fields, "Non Ground points") for cloud in n_ground_csf]

    #union de nubes de puntos
    #ground points
    merged_ground_points= mergePointClouds(cc_ground_clouds, name_merge='Ground points')
    merged_ground_points.setCurrentDisplayedScalarField(merged_ground_points.getScalarFieldIndexByName("Zcoord"))

    #no ground points
    merged_n_ground_points= mergePointClouds(cc_n_ground_clouds, name_merge='Non Ground points')
    merged_n_ground_points.setCurrentDisplayedScalarField(merged_n_ground_points.getScalarFieldIndexByName("Zcoord"))

    #resultado a cloudcompare
    cc.addToDB(merged_ground_points)
    cc.addToDB(merged_n_ground_points)
    cc.updateUI()

    #====================================================
    #Para visualizar las celdas en CloudCompare
    #====================================================
    grid = pycc.ccPointCloud()
    for cell in cells:
        for point in cell:
            grid.addPoint(cccorelib.CCVector3(point[0], point[1], 0))
    grid.setName("Grid")
    cc.addToDB(grid)
    #====================================================
    #====================================================

    del cc, point_cloud, random_subsample_cloud, denoised_cloud, xyz, clouds, ground_csf, n_ground_csf, ground_espectral

    print("PROCESO FINALIZADO")


