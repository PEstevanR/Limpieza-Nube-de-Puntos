
#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================
"""
Objetivo:
        Obtener los puntos que hacen parte de cada carril de via como paso previo a la obtención del eje de via
Proceso:
        1-)Cargamos los datos como ndarray(x,y,z) desde cloudcompare

        2-)Obtenemos nubes de puntos por celdas de tamaño 'cell_size' de todo el conjunto de datos

        3-)Filtramos los puntos acorde a varias condiciones para obtener puntos potenciales de ser de carril

        4-)Ajustamos localmente con RANSAC para obtener inliers

        5-)reconstrucción de railes y eje de via (...en proceso de desarrollo)

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

#modulos de cloudcompare y otros
import pycc
import cccorelib
import numpy as np

#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, mergePointClouds
from utils.utilities import getTrackPoints, customRansac


#Aqui inicia la ejecución del código
if __name__=="__main__":
#1-)
    cc, _ , xyz = getCloud()
#2-)
    cells, clouds = getCellPointCloud(xyz, cell_size=1)
#3-)
    track_clouds = getTrackPoints(clouds)

#4-)Ajustamos localmente con RANSAC para obtener inliers
    clouds =[]
    for track_points in track_clouds:
        #Ajuste RANSAC
        inliers, outliers, m = customRansac(track_points)
        #Si los puntos inliers del modelo RANSAC son más del 90% nos quedamos con ellos
        condition4 = (inliers.shape[0] / track_points.shape[0])*100
        if condition4 > 90.0:
            track_points = track_points[inliers]
            direction = np.full(shape=(track_points.shape[0]), fill_value=m)
            track_points = np.append(track_points, np.expand_dims(direction, axis=1), 1)

            pc= pycc.ccPointCloud(track_points[:,0], track_points[:,1], track_points[:,2])
            pc.addScalarField("RANSAC Direction", track_points[:,3])
            pc.setName("Puntos ground por celda")
            pc.setCurrentDisplayedScalarField(pc.getScalarFieldIndexByName("RANSAC Direction"))
            clouds.append(pc)

    #Para visualizar en cloudcompare
    merged_cloud = mergePointClouds(clouds, name_merge='RANSAC_inliers')
    merged_cloud.setCurrentDisplayedScalarField(merged_cloud.getScalarFieldIndexByName("RANSAC_direction"))
    cc.addToDB(merged_cloud)


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

#5-)reconstrucción de railes y eje de via (...en proceso de desarrollo)

    print("PROCESO FINALIZADO")



