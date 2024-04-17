

import pycc
import cccorelib
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN


"""
Objetivo:
        Filtrar los puntos que son paralelos entre sí por vecindarios de 1.7m (ancho de via + 0.1m)
Proceso:
        1-) carga de la nube de puntos desde cloudcompare
        2-) kdtree y selección de puntos cuyos vecindarios son paralelos en un 80% como mínimo

Resultado:
        No se consigue filtrar totalmente los puntos de carril. Buscar otras alternativas como la transformada de hough

"""

def main():
    cc = pycc.GetInstance() # pycc.ccPythonInstance object
#1-)
    #CloudCompare point cloud to ndarray
    entities = cc.getSelectedEntities() #ccPointCloud objects list
    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    point_cloud = entities[0] #ccPointCloud object
    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    xyz_CloudCompare = point_cloud.points() #Original point cloud (x y z) as numpy array not writeable
    sf = point_cloud.getScalarField(point_cloud.getScalarFieldIndexByName('RANSAC_direction')).asArray()
    xyz_CloudCompare = np.append(xyz_CloudCompare, np.expand_dims(sf, axis=1), 1)

    #copy from oribinal point cloud
    xyz = xyz_CloudCompare.copy()
    xyz.flags.writeable = True #ndarray writeable

#2-)
    # Crear KDTree
    tree = KDTree(xyz[:, :2], leaf_size=50)

    pto = pycc.ccPointCloud()
    pto.setName("PUNTO")
    for i in range(xyz.shape[0]):
        #if i ==80000:
        point = xyz[i]
        neighbors_idx = tree.query_radius(point[:2].reshape(1,-1), r=1.6 + 0.1)[0]
        near_points = xyz[neighbors_idx]
        near_points_direction = near_points[:,3]

        umbral_de_similitud = 0.001  # Ajusta este umbral según sea necesario
        direcciones_similares = np.abs(near_points_direction - point[3]) < umbral_de_similitud
        porcentaje = np.count_nonzero(direcciones_similares) / near_points.shape[0] * 100

        if porcentaje > 80.0:
            print("Porcentaje de puntos cercanos con dirección similar:", porcentaje)
            pto.addPoint(cccorelib.CCVector3(point[0], point[1], point[2]))
    cc.addToDB(pto)

if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO ")