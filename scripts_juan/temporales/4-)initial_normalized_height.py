

import pycc
import numpy as np
import dendromatics as dm

"""
Objetivo:
        Obtener una nube de puntos normalizada
Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare
        2-)Normalización de altura mediante un DTM temporal
        3-)Selección de los puntos con menos de 50cm de altura
        4-)Nube normalizada a CloudCompare

Resultado:
        Se obtiene la nube de puntos normalizada con aquellos puntos por debajo de 50cm.

"""


def main():
    #Para interactuar con los datos seleccionados en CloudCompare, devuelve un objeto pycc.ccPythonInstance()
    cc = pycc.GetInstance()
#1-)
#de CloudCompare a objeto ccPointCloud
    entities = cc.getSelectedEntities() #Lista de objetos ccPointCloud
    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    point_cloud = entities[0] #primer objeto ccPointCloud
    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    #de objeto ccPointCloud a ndarray(x y z) numpy
    xyz_CloudCompare = point_cloud.points()

    #copia de la nube de puntos original
    xyz = xyz_CloudCompare.copy()
    xyz.flags.writeable = True #ndarray writeable

#2-)
#Normalización de alturas
    clean_points = dm.clean_ground(xyz)
    #generar DTM
    cloth_nodes = dm.generate_dtm(clean_points)
    #limpiando el DTM
    dtm = dm.clean_cloth(cloth_nodes)
    #interpolacion de valores faltantes
    complete_dtm = dm.complete_dtm(dtm)
    #Normalización de alturas
    z0 = dm.normalize_heights(xyz, complete_dtm)
    xyz = np.append(xyz, np.expand_dims(z0, axis=1), 1)

#3-)
# Selección de puntos por debajo de 0.5m
    stripe = xyz[xyz[:, 3] < 0.5, 0:4]

#4-)
#Nube de puntos voxelizada a cloudCompare
    x = stripe[:,0]
    y = stripe[:,1]
    z = stripe[:,3]
    voxelate_point_cloud = pycc.ccPointCloud(x,y,z)
    voxelate_point_cloud.setName(point_cloud.getName() + "- Voxelated")

    #scalarfield
    voxelate_point_cloud.addScalarField("Z_normalizada", z)
    voxelate_point_cloud.setCurrentDisplayedScalarField(voxelate_point_cloud.getScalarFieldIndexByName("Z_normalizada"))
    cc.addToDB(voxelate_point_cloud)
    cc.updateUI()

if __name__=="__main__":
    main()
    print("FINALIZADO EL PROCESO")