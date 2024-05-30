

import pycc
import cccorelib
import numpy as np

"""
Objetivo:
        Asignar a cada punto una dirección derivada de aplicar PCA a cada celda de una malla de tamaño cell_size
Proceso:
        1-)Cargamos los datos como ndarray(x,y,z) desde cloudcompare

        2-)Obtenemos un mallado de tamaño 'cell_size' en todo el conjunto de datos

        3-)Aplicamos PCA al conjunto de puntos contenidos en cada celda de la malla y asignamos la direccion a cada punto
        como la direccion de máxima variación de los datos (direccion del autovector del CP1)

Resultado:
        -se consigue el objetivo propuesto, pero pueden existir problemas de calculo a la hora de aplicar PCA
        debido a que entremedias se calcula la matriz de covarianzas y si no hay los datos sufientes, se lanza una excepción.

        -se continua con la aplicación de RANSAC para comparar resultados.

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





#Calculo de PCA
def customPCA(cloud):
    #PCA
    m = np.mean(cloud, axis=0)
    cloud_norm = cloud - m
    cov = np.cov(cloud_norm.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov.T)
    sorted_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:, sorted_indexes]
    return eigenvectors[:2,:2] #cp1 y cp2





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
    cells = cell_grid(xyz, cell_size=1)

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
                eig_vec = customPCA(cell_points)

                #calculo de la dirección del primer vector propio
                direccion_radianes = np.arctan2(eig_vec[0][1], eig_vec[0][0]) #tan^-1(y/x)
                direccion_grados = np.degrees(direccion_radianes)

                #=============================================================
                #Para visualizar las nubes de punto por celdas en CloudCompare
                #=============================================================
                direccion = np.full(shape=(cell_points.shape[0],1), fill_value=direccion_grados) #to add ScalarField

                points = pycc.ccPointCloud(cell_points[:,0], cell_points[:,1], cell_points[:,2])
                points.addScalarField("PCA_direction", direccion[: , 0])
                clouds.append(points)
                #=============================================================
                #=============================================================

    merged_cloud = merge(clouds, name_merge='PCA_direction')
    merged_cloud.setCurrentDisplayedScalarField(merged_cloud.getScalarFieldIndexByName("PCA_direction"))
    cc.addToDB(merged_cloud)

if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO ")