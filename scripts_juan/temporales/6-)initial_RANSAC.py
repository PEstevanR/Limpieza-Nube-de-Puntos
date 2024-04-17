

import pycc
import cccorelib
import numpy as np
from sklearn.linear_model import RANSACRegressor

"""
Objetivo:
        -Filtrar los puntos que se ubican a 3cm del mejor ajuste RANSAC por m^2, para obtener los puntos de carril
        -asignar a cada punto filtrado la pendiente de la recta de regresion como direccion
Proceso:
        1-)Cargamos los datos como ndarray(x,y,z) desde cloudcompare

        2-)Obtenemos un mallado de tamaño 'cell_size' en todo el conjunto de datos

        3-)Aplicamos RANSAC al conjunto de puntos contenidos en cada celda de la malla y seleccionamos los inliers

Resultado:
        Se alcanzan a seleccionar ciertos puntos que son de carril, pero sigue existiendo ruido.
        Mejores resultados que con PCA

"""

#Obtención de grid
def cell_grid(cloud, cell_size):
    #Init
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




def custom_ransac(cloud):
    # Extraer las coordenadas x, y, z
    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]

    # Aplicar la regresión RANSAC
    model_ransac = RANSACRegressor(min_samples=2,
                                    max_trials=100,
                                    loss='absolute_error',
                                    residual_threshold=0.03,
                                    stop_probability=0.99,
                                    random_state=123)
    #x = x*z
    model_ransac.fit(x[:, np.newaxis], y)

    # Obtener los puntos que se ajustan al modelo
    inliers = model_ransac.inlier_mask_
    outliers = np.logical_not(inliers)
    return inliers, outliers, model_ransac.estimator_.coef_[0]




def merge(clouds, name_merge='merge_cloud'):
    total_num_points = sum(cloud.size() for cloud in clouds)

    merge_result = pycc.ccPointCloud(name_merge)
    merge_result.reserve(total_num_points)

    for cloud_idx, cloud in enumerate(clouds):
        for point_idx in range(cloud.size()):
            merge_result.addPoint(cloud.getPoint(point_idx))

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

    #copy from oribinal point cloud
    xyz = xyz_CloudCompare.copy()
    xyz.flags.writeable = True #ndarray writeable

#2-)
    cells = cell_grid(xyz, cell_size=0.5)

    #====================================================
    #Para visualizar las celdas en CloudCompare como puntos
    #====================================================
    grid = pycc.ccPointCloud()
    for cell in cells:
        for point in cell:
            grid.addPoint(cccorelib.CCVector3(point[0], point[1], 0))
    grid.setName("Mallado")
    cc.addToDB(grid)
    #====================================================
    #====================================================

#3-)
    #Iterando por celda
    clouds = []
    for cell in cells:

        # Limites de cada celda
        x_min, y_min = cell[0]
        x_max, y_max = cell[2]

        # filtrando los puntos que estan contenidos en cada celda
        mask = np.where((xyz[:,0] >= x_min) & (xyz[:,0] <= x_max) & (xyz[:,1] >= y_min) & (xyz[:,1] <= y_max))
        cell_points = xyz[mask]

        #Trabajamos con celdas que tengan mas de 20 puntos (esto puede cambiarse)
        if cell_points.shape[0] > 20:
                #Ajuste RANSAC
                inliers, outliers, m = custom_ransac(cell_points)

                condition1 = (inliers.shape[0] / cell_points.shape[0])*100
                if condition1 > 90.0:

                    x = cell_points[:, 0]
                    y = cell_points[:, 1]
                    z = cell_points[:, 2]

                    pc_inliers = pycc.ccPointCloud(x[inliers], y[inliers], z[inliers])

                    direction = np.full(shape=(x[inliers].shape[0],1), fill_value=m) #to add ScalarField

                    pc_inliers.addScalarField("RANSAC_direction", direction[: , 0])
                    clouds.append(pc_inliers)

    merged_cloud = merge(clouds, name_merge='RANSAC_inliers')
    merged_cloud.setCurrentDisplayedScalarField(merged_cloud.getScalarFieldIndexByName("RANSAC_direction"))
    cc.addToDB(merged_cloud)


if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO ")