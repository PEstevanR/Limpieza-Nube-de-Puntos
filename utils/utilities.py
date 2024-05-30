
import CSF
import pycc
import numba
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import RANSACRegressor

#==============================================================================================================
#============================================De proposito general==============================================
#==============================================================================================================
#obtener nube de puntos desde cloudcompare
#@numba.njit no compatible con pycc
def getCloud():
    """
    _summary_:
        Permite leer la nube de puntos seleccionada en CloudCompare y obtener el objeto cc para interactuar con CloudCompare
    Args:

    Returns:
        cc (pycc.ccPythonInstance()): objeto para interactuar con CloudCompare
        point_cloud(pycc.ccPointCloud): nube de puntos como objeto de CloudCompare
        cloud_ndarray(array): array numpy de la forma (x,y,z)
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
    cloud_ndarray = xyz_CloudCompare.copy()
    cloud_ndarray.flags.writeable = True #ndarray writeable
    return cc, point_cloud, cloud_ndarray




#@numba.njit no compatible con pycc
def makePyccPointCloudObject(cloud, scalar_fields, name='cloud'):
    """
    _summary_:
        Permite convertir un array numpy a un objeto ccPointCloud
    Args:
        cloud (ndarray(x,y,z)): nube de puntos como array numpy
        scalar_fields(list): nombre de los campos escalares
        name(str, optional): nombre de la nube de puntos.
    Returns:
        point_cloud(ccPointCloud()): nube de puntos resultado de la unión
    """
    x = cloud[:,0]
    y = cloud[:,1]
    z = cloud[:,2]
    point_cloud = pycc.ccPointCloud(x,y,z)
    point_cloud.setName(name)
    point_cloud.addScalarField("Zcoord", z)
    point_cloud.setCurrentDisplayedScalarField(point_cloud.getScalarFieldIndexByName("Zcoord"))

    #añadiendo los demas campos escalares
    if cloud.shape[1]>3:
        if len(scalar_fields)>0:
            for idx, field_name in enumerate(scalar_fields):
                point_cloud.addScalarField(field_name, cloud[:,idx+3])
    return point_cloud



#Obtención de grid (mallado)
@numba.njit
def makeGrid(cloud, cell_size):
    """
    _summary_:
        Permite obtener un grid de paso=cell_size que abarca la extension total de la nube de puntos en el plano 2D
    Args:
        cloud (ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda
    Returns:
        cells(array): lista de arrays donde cada array contiene los cuatro puntos que definen una celda.
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



#obtener nube de puntos por celda
#@numba.njit no compatible con cKDTree
def getCellPointCloud(xyz, cell_size):
    """
    _summary_:
        Obtener nube de puntos por celda, la busqueda de vecinos se realiza mediante arbol KDTree
    Args:
        xyz(ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda
    Returns:
        cells(array): lista de arrays donde cada array contiene los cuatro puntos que definen una celda.
        clouds(list): lista de nubes de puntos por celda
    """
    #obtención de grid
    cells = makeGrid(xyz, cell_size)

    #Arbol KD en el plano 2D
    tree = cKDTree(xyz[:,:2])

    #lista de nubes de puntos (una por celda)
    clouds = []
    for cell in cells:
        # Limites de cada celda
        x_min, y_min = cell[0]
        x_max, y_max = cell[2]
        # filtrado de puntos que estan contenidos en la circunferencia que inscribe a cada celda
        d = ((cell_size/2)**2 + (cell_size/2)**2)**0.5
        indices = tree.query_ball_point([(x_min + x_max) / 2, (y_min + y_max) / 2], d)
        cell_points = xyz[indices]
        clouds.append(cell_points)
    return cells, clouds



#Union de nubes de puntos
#@numba.njit no compatible con pycc
def mergePointClouds(clouds, name_merge='merge_cloud'):
    """
    _summary_:
        Union de nubes de puntos
    Args:
        clouds (list): lista de nubes de puntos como objetos pycc.ccPointCloud
        name_merge (str, optional): nombre de la nube de puntos resultado.
    Returns:
        merge_result(ccPointCloud()): nube de puntos resultado de la unión
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






#==============================================================================================================
#================================================Rail detection================================================
#==============================================================================================================
#Análisis de componentes principales (PCA)
@numba.njit
def customPCA(cloud):
    """
    _summary_:
        Aplica Análisis de Componentes Principales (PCA) a la nube de puntos
    Args:
        cloud (ndarray(x,y,z)): nube de puntos como array numpy
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


#Análisis de regresión lineal RANSAC
#@numba.njit No compatible con RANSACRegressor()
def customRansac(cloud):
    """
    _summary_:
        Regresion RANSAC para determinar puntos colineales basada en vecindarios
    Args:
        cloud (ndarray(x,y,z)): nube de puntos
    Returns:
        inliers(array): máscara de puntos que estan sobre la recta de regresión
        outliers(array): máscara de puntos que no estan sobre la recta de regresión
        m(float): pendiente de la recta de regresión
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
    m = model_ransac.estimator_.coef_[0]
    return inliers, outliers, m




#Obtener puntos potenciales de ser de carril en una nube de puntos
@numba.njit
def getTrackPoints(clouds):
    """
    _summary_:
        Filtrar los puntos de carril presentes en una nube de puntos
    Args:
        clouds(array): lista con nubes de puntos ndarray(x,y,z)
    Returns:
        track_clouds(list): lista con nubes de puntos (ndarray(x,y,z) potencialmente de carril
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








#==============================================================================================================
#=====================================================RPAS=====================================================
#==============================================================================================================

#Filtro (geométrico) de puntos de terreno natural con Cloth Simulation Filter (CSF)
#@numba.njit no compatible con pycc
def filterCSF(clouds, params):
    """
    _summary_:
        Filtrado geométrico de puntos de terreno natural con el algoritmo Cloth Simulation Filter
    Args:
        clouds (list): lista de nubes de puntos ndarray(x,y,z)
        params (list): lista de parametros [bSloopSmooth, class_threshold, cloth_resolution, rigidness, interations]
    Returns:
        ground_point_clouds(list): lista nube de puntos (ndarray) de terreno natural
        n_ground_point_clouds(list): lista de nubes de puntos (ndarray) de objetos sobre el terreno natural como
    """
    #algoritmo CSF
    csf = CSF.CSF()

    #parametros del algoritmo
    csf.params.bSloopSmooth = params[0]    # manejar pendientes pronunciadas después de la simulación
    csf.params.class_threshold = params[1] # umbral de clasificación puntos terreno y no terreno
    csf.params.cloth_resolution = params[2]# resolución de la tela
    csf.params.rigidness = params[3]       # 1)montaña y vegetacion densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    csf.params.interations = params[4]     # numero de iteraciones

    #filtramos por celdas
    ground_point_clouds = []
    n_ground_point_clouds = []
    for cloud in clouds:
        #nubes de puntos con menos de 10 puntos las saltamos
        if cloud.shape[0] < 10:
            continue
        csf.setPointCloud(cloud)
        idx_ground = CSF.VecInt()             # una lista para indicar el índice de puntos de tierra después del cálculo
        idx_n_ground = CSF.VecInt()           # una lista para indicar el índice de puntos no terrestres después del cálculo
        csf.do_filtering(idx_ground, idx_n_ground)# filtrado

        #Indices de puntos terreno y no terreno
        idx_ground_array = np.array(idx_ground).astype(int)
        idx_n_ground_array = np.array(idx_n_ground).astype(int)

        # extracción de puntos terreno
        if idx_ground_array.shape[0] > 0:
            ground_points = cloud[idx_ground_array]
            ground_point_clouds.append(ground_points)

        # extraccion de puntos sobre el terreno
        if idx_n_ground_array.shape[0] > 0:
            n_ground_points = cloud[idx_n_ground_array]
            n_ground_point_clouds.append(n_ground_points)

    return ground_point_clouds, n_ground_point_clouds





#Calculo de indices espectrales con campos escalares RGB
#@numba.njit no compatible con pycc
def espectralInfo(cloud):
    """
    _summary_:
        Obtención de campos escalares e indices espectrales de una nube de puntos
    Args:
        cloud(pycc.ccPointCloud): nube de puntos como objeto de CloudCompare
    Returns:
        xyz(array): array numpy de la forma (x,y,z, scalarfields)
        aux_dict.keys()(array): nombre de campos escalares
    """
    #se pasa la nube de puntos a un array de numpy con todos sus campos escalares
    xyz = cloud.points() #ndarray

    #recuperando los valores de los campos escalares
    scalar_fields_names = [cloud.getScalarFieldName(i) for i in range(cloud.getNumberOfScalarFields())]
    aux_dict = {}
    for idx, field_name in enumerate(scalar_fields_names):
        if field_name not in aux_dict.keys():
            aux_dict[field_name] = cloud.getScalarField(idx).asArray()

    #recuperando color en el espacio RGB
    if cloud.colors() is not None:
        aux_dict['Red'],aux_dict['Green'],aux_dict['Blue'], _ = cloud.colors().T

        #============================Añadiendo indices espectrales============================
        #negativo banda roja
        aux_dict['Negative_R'] = 255-aux_dict['Red']

        #Index Excess Green
        aux_dict['ExG'] =(2*aux_dict['Green']-aux_dict['Red']-aux_dict['Blue'])

        #Visible Normalized Difference Vegetation Index
        aux_dict['vNDVI'] = 0.5268*aux_dict['Red']**-0.1294 * aux_dict['Green']**0.3389 * aux_dict['Blue']**-0.31118

        #brillo
        #aux_dict['Brightness'] = aux_dict['Red'] + aux_dict['Green'] + aux_dict['Blue']

        #Color Index of Vegetation
        #aux_dict['CIVE'] = 0.441*aux_dict['Red'] - 0.881*aux_dict['Green'] + 0.385*aux_dict['Blue'] + 18.78745

        #Index Excess Green Minus Excess Red
        #aux_dict['EXG'] =(2*aux_dict['Green']-aux_dict['Red']-aux_dict['Blue']) - (1.4*aux_dict['Red']-aux_dict['Green'])

        #Green Leaf Index
        #numerador = 2*aux_dict['Green'] - aux_dict['Red'] - aux_dict['Blue']
        #denominador = 2*aux_dict['Green'] + aux_dict['Red'] + aux_dict['Blue']
        #aux_dict['GLI'] = np.where(denominador != 0, numerador / denominador, np.nan)

        #Soil Adjusted Vegetation Index
        #numerador = 1.5*(aux_dict['Green']-aux_dict['Red'])
        #denominador = aux_dict['Green']+aux_dict['Red']+0.5
        #aux_dict['SAVI'] = np.where(denominador != 0, numerador / denominador, np.nan)

        #Visual Atmospheric Resistance Index
        #numerador = aux_dict['Green']-aux_dict['Red']
        #denominador = aux_dict['Green']+aux_dict['Red']-aux_dict['Blue']
        #aux_dict['VARI'] = np.where(denominador != 0, numerador / denominador, np.nan)

    #añadiendo todo al ndarray
    for key, values in aux_dict.items():
        lista_array = np.array(values)
        xyz = np.column_stack((xyz, lista_array))
    return xyz, aux_dict.keys()


def espectralFilter():
    pass

def interpolateContours(cloud):
    pass