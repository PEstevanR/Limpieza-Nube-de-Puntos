
#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================

import pycc
import numba
import cccorelib
import numpy as np
import dendromatics as dm
from sklearn.linear_model import RANSACRegressor

"""
Objetivo:
        Obtener los puntos que hacen parte de cada carril de via
Proceso:
        1-)Cargamos los datos como ndarray(x,y,z) desde cloudcompare

        2-)Obtenemos nubes de puntos por celdas de tamaño 'cell_size' de todo el conjunto de datos

        3-)Filtramos los puntos acorde a varias condiciones para obtener puntos potenciales de ser de carril

        4-)Ajustamos localmente con RANSAC para obtener inliers

        5-)

Resultado:
        -Se consigue filtrar los puntos que pertencen a las vias, pero sigue existiendo ruido
        -Debe mejorarse el filtrado de puntos por celda (el proceso aun es lento)

"""

#Obtención de grid
@numba.njit
def makeGrid(cloud, cell_size):
    """
    _summary_:
        Permite obtener un grid de paso=cell_size que abarca la extension total de la nube de puntos en el plano 2D
    Args:
        cloud (ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda
    Returns:
        cells(array): Cada array contiene los cuatro puntos que definen una celda.
    """
    # Init
    x = cloud[:,0]
    y = cloud[:,1]

    # número de celdas en x e y
    n_cell_x = int(np.ceil((x.max() - x.min()) / cell_size))
    n_cell_y = int(np.ceil((y.max() - y.min()) / cell_size))

    # límites de las celdas
    cell_x = np.linspace(x.min(), x.max(), n_cell_x + 1)
    cell_y = np.linspace(y.min(), y.max(), n_cell_y + 1)

    # array para almacenar los puntos que definen cada celda
    cells= np.empty((n_cell_x * n_cell_y, 4, 2))
    count=0
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            x_min = cell_x[i]
            x_max = cell_x[i + 1]
            y_min = cell_y[j]
            y_max = cell_y[j + 1]
            cell_points = np.array([[x_min, y_min],
                                    [x_max, y_min],
                                    [x_max, y_max],
                                    [x_min, y_max]
                                    ])
            cells[count] = cell_points
            count+=1
    return cells

#obtencion de inliers, outliers y dirección (RANSAC)
#@numba.njit No compatible con RANSACRegressor()
def custom_ransac(cloud):
    """
    _summary_:
        Regresion RANSAC para determinar el mejor ajuste de una linea recta sin influencia de outliers
    Args:
        cloud (ndarray(x,y,z)): nube de puntos
    Returns:
        inliers(array): ids de inliers
        outliers(array): id de outliers
        m(float): pendiente (dirección) de la recta de regresión
    """
    # coordenadas x, y, z
    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]

    # regresión RANSAC
    model_ransac = RANSACRegressor(min_samples=2,
                                    max_trials=100,
                                    loss='absolute_error',
                                    residual_threshold=0.03,
                                    stop_probability=0.99,
                                    random_state=123)
    model_ransac.fit(x[:, np.newaxis], y)

    # identificadores de inliers y outliers
    inliers = model_ransac.inlier_mask_
    outliers = np.logical_not(inliers)
    m =model_ransac.estimator_.coef_[0]
    return inliers, outliers, m


#Union de nubes de puntos
#@numba.njit no compatible con pycc
def mergePointClouds(clouds, name_merge='merge_cloud'):
    """
    _summary_:
        Union de nubes de puntos
    Args:
        clouds (list): lista de nubes de puntos en formato ndarray(x, y, z)
        name_merge (str, optional): nombre de la nube de puntos resultado.
    Returns:
        merge_result(objeto ccPointCloud()): nube de puntos resultado de la union
    """
    # nueva nube de puntos
    total_num_points = sum(cloud.size() for cloud in clouds)
    merge_result = pycc.ccPointCloud(name_merge)
    merge_result.reserve(total_num_points)

    # añadir cada punto a la nueva nube de puntos
    for cloud_idx, cloud in enumerate(clouds):
        for point_idx in range(cloud.size()):
            merge_result.addPoint(cloud.getPoint(point_idx))

    #configuración del campo escalar
    pos = 0
    for cloud in clouds:
        for i in range(cloud.getNumberOfScalarFields()):
            scalarFieldName = cloud.getScalarFieldName(i)
            idx = merge_result.getScalarFieldIndexByName(scalarFieldName)
            if idx == -1:
                idx = merge_result.addScalarField(scalarFieldName)
                if idx == -1:
                    raise RuntimeError("Failed to add ScalarField")
            scalarField = cloud.getScalarField(i)
            sf = merge_result.getScalarField(idx)
            sf.asArray()[pos: pos + scalarField.size()] = scalarField.asArray()[:]
            sf.computeMinAndMax()
        pos += cloud.size()
    return merge_result



#obtener nube de puntos desde cloudcompare
#@numba.njit no compatible con pycc
def getCloud():
    """
    _summary_:
        Obtener nube de puntos como ndarray writeable e instancia de objeto para interactuar con cloudcompare
    Args:
        None
    Returns:
        cc (objeto pycc.GetInstance()): Para interactuar con CloudCompare,
        xyz(ndarray(x,y,z)): nube de puntos
    """
    # para interactuar con los datos seleccionados en CloudCompare, devuelve un objeto pycc.ccPythonInstance()
    cc = pycc.GetInstance()

    # de CloudCompare a objeto ccPointCloud
    entities = cc.getSelectedEntities() #Lista de objetos ccPointCloud
    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    point_cloud = entities[0] # primer objeto ccPointCloud
    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    # de objeto ccPointCloud a ndarray(x y z) numpy
    xyz_CloudCompare = point_cloud.points()

    # copia de la nube de puntos original
    xyz = xyz_CloudCompare.copy()
    xyz.flags.writeable = True #ndarray writeable
    return cc, xyz



#obtener nube de puntos por celda
@numba.njit
def getCellPointCloud(xyz):
    """
    _summary_:
        Obtener nube de puntos por celda
    Args:
        xyz(ndarray(x,y,z)): nube de puntos
    Returns:
        cells(array): lista de arrays donde cada array contiene los cuatro puntos que definen una celda.
        clouds(list): lista de nubes de puntos por celda
    """
    cells = makeGrid(xyz, cell_size=1) # con cell_size= 1 funciona bastante bien, aunque tarda más
    #lista de nubes de puntos (una por celda)
    clouds = []
    for cell in cells:
        # Limites de cada celda
        x_min, y_min = cell[0]
        x_max, y_max = cell[2]
        # filtrando los puntos que estan contenidos en cada celda
        mask = np.where((xyz[:,0] >= x_min) & (xyz[:,0] <= x_max) & (xyz[:,1] >= y_min) & (xyz[:,1] <= y_max))
        cell_points = xyz[mask]
        clouds.append(cell_points)
    return cells, clouds



#obtener percentiles
@numba.njit
def getPercentil(data, percentil):
    """
    _summary_:
        Obtener un determinado percentil de una lista de valores numéricos
    Args:
        data(array): valores numéricos
        percentil(int, float): valor percentil
    Returns:
        value(int, float): valor correspondiente al percentil
    """
    data_sorted = np.sort(data)
    index = int(np.ceil((percentil / 100.0) * len(data_sorted))) - 1
    value = data_sorted[index]
    return value



#Obtener puntos de carril de una nube de puntos
@numba.njit
def getTrackpoints(clouds):
    """
    _summary_:
        Filtrar los puntos de carril presentes en una nube de puntos
    Args:
        clouds(array): lista con nubes de puntos ndarray(x,y,z)
    Returns:
        track_clouds(list): lista con puntos potencialmente de carril
    """
    # init
    track_clouds = []
    for cell_points in clouds:
        #Trabajamos con celdas que tengan mas de 20 puntos (esto puede cambiarse)
        if cell_points.shape[0] > 20:
            #Percentil 10% de altura en cada celda
            mdt = getPercentil(cell_points[:, 2], 10)

            ##Condicion 1
            #Verificacion de que menos del 10% de puntos se situen entre mdt+0.5 y mdt+4.5
            mask_1 = np.where((cell_points[:, 2] >= mdt+0.5) & (cell_points[:, 2] <= mdt+4.5))
            condition1 = (cell_points[mask_1].shape[0] / cell_points.shape[0])*100 #Porcentaje de puntos entre [mdt+0.5 , mdt+4.5]
            if condition1 < 10.0:
                #En las celdas que cumplen la condicion, se filtran los puntos del terreno (z < mdt+0.5)
                mask_2 = np.where((cell_points[:, 2] < mdt+0.5))
                ground_points = cell_points[mask_2]

            ##Condicion 2
                #Percentiles de altura
                p10 = getPercentil(ground_points[:, 2], 10)
                p98 = getPercentil(ground_points[:, 2], 98)
                condition2 = p98-p10
                #Si la diferencia de percentiles es mayor a 10cm, los ground_points se filtran para obtener los puntos
                #que estan entre (p98-0.10 , p98)
                if condition2 > 0.10:
                    mask_3 = np.where((ground_points[:, 2] > p98-0.10) & (ground_points[:, 2] < p98))
                    track_points = ground_points[mask_3]

            ##Condicion 3
                    #Si los puntos que estan entre (p98-0.10 , p98) son menos del 50% respecto a los ground_points,
                    #estos puntos son potencialemnte puntos de riel
                    condition3 = (track_points.shape[0] / ground_points.shape[0])*100
                    if condition3 < 50:
                        #nos quedamos con las celdas con mas de 20 puntos
                        if track_points.shape[0] > 20:
                            track_clouds.append(track_points)
    return track_clouds


#Aqui inicia la ejecución del código
if __name__=="__main__":
#1-)
    cc, xyz = getCloud()
#2-)
    cells, clouds = getCellPointCloud(xyz)
#3-)
    track_clouds = getTrackpoints(clouds)

#Para visualizar en cloudcompare
    clouds =[]
    for track_points in track_clouds:
        #Ajuste RANSAC
        inliers, outliers, m = custom_ransac(track_points)
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
    print("PROCESO FINALIZADO")



