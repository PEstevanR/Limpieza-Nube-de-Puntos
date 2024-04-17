import pycc
import numpy as np
from sklearn.linear_model import RANSACRegressor


def cell_grid(cloud, cell_size):
    """
    _summary_:
        Permite obtener un grid de paso=cell_size que abarca la extension total de la nube de puntos en el plano 2D

    Args:
        cloud (ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda

    Returns:
        lista de arrays: Cada array contiene los cuatro puntos que definen una celda.
    """
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




def customPCA(cloud):
    """
    _summary_:
        Aplica Análisis de Componentes Principales a la nube de puntos (PCA)

    Args:
        cloud (ndarray(x,y,z)): nube de puntos

    Returns:
        eigenvectors: los primeros dos autovectores del análisis
    """

    #PCA
    m = np.mean(cloud, axis=0)
    cloud_norm = cloud - m
    cov = np.cov(cloud_norm.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov.T)
    sorted_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:, sorted_indexes]
    return eigenvectors[:2,:2] #cp1 y cp2



def custom_ransac(cloud):
    """
    _summary_:
        Regresion RANSAC para determinar la dirección de un punto basada en vecindarios

    Args:
        cloud (ndarray(x,y,z)): nube de puntos

    Returns:
        mascaras de inliers y outliers del ajuste. Ademas, la pendiente (dirección) de la recta de regresión
    """
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
    """
    _summary_:
        Union de nubes de puntos

    Args:
        clouds (list): lista de nubes de puntos en formato ndarray(x, y, z)
        name_merge (str, optional): nombre de la nube de puntos resultado.

    Returns:
        nube de puntos resultado de la union
    """

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