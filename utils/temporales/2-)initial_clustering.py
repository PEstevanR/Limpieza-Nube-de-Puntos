

import pycc
from sklearn.cluster import DBSCAN

"""
Objetivo:
        Probar el funcionamiento del algoritmo DBSCAN y aplicarlo a la nube de puntos para detectar ruido global.

Proceso:
        1-)Carga la nube de puntos seleccionada en CloudCompare
        2-)Aplica clustering (DBSCAN)
        3-)Añadir campo escalar a la nube de puntos con id de clusters.

Resultado:
        Proceso muy lento mirar otras alternativas
"""


def main():
    #Para interactuar con los datos seleccionados en CloudCompare, devuelve un objeto pycc.ccPythonInstance()
    cc = pycc.GetInstance()
#1-)
#de CloudCompare a objeto ccPointCloud

    entities = cc.getSelectedEntities() #Lista de objetos ccPointCloud
    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    cloud = entities[0] # primer objeto ccPointCloud
    if not isinstance(cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    #de objeto ccPointCloud a ndarray(x y z) numpy
    xyz = cloud.points()


#2-)
#DBSCAN
    d=1 #máxima distancia entre 2 puntos para ser agrupados (probar con otros valores)
    min_points = 50 #minimo de puntos para formar un grupo  (probar con otros valores)
    dbscan = DBSCAN (eps=d, min_samples=min_points, n_jobs=-1).fit(xyz) #clustering
    clusters = dbscan.labels_ #id de clusters

#3-)
#Añadir campo escalar a la nube de puntos
    scalar_fields: list[str] = []
    for i in range(cloud.getNumberOfScalarFields()):
        scalar_fields.append(cloud.getScalarFieldName(i))

    #campo escalar de nombre DBSCAN_clusters
    if 'DBSCAN_clusters' not in scalar_fields:
        idx=cloud.addScalarField("DBSCAN_clusters", clusters)
    else:
        raise RuntimeError("ScalarField 'DBSCAN_clusters' already exists")

    cc.updateUI()


if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO")