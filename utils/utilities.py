
import CSF
import pycc
import numba
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RANSACRegressor

"""
Utilities.py es la biblioteca de métodos o funciones que se han ido desarrollando
para la limpieza de nubes de puntos de ferrocarril y RPAS; con el objetivo de luego
pasar todas las funcionalidades a una programación orientada a objetos.
"""
#==============================================================================================================
#=======================================Funciones de proposito general=========================================
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
        raise RuntimeError("Seleccione una nube de puntos")

    point_cloud = entities[0] # primer objeto ccPointCloud
    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("La entidad seleccionada debe ser una nube de puntos")

    # de objeto ccPointCloud a ndarray(x y z) numpy
    xyz_CloudCompare = point_cloud.points()

    # copia de la nube de puntos original
    cloud_ndarray = xyz_CloudCompare.copy()
    cloud_ndarray.flags.writeable = True #ndarray writeable
    return cc, point_cloud, cloud_ndarray



#@numba.njit no compatible con pycc
def getCloudAsArray(cloud):
    """
    _summary_:
        Obtención de la nube de puntos como un ndarray numpy con todos los campos escalares
    Args:
        cloud(pycc.ccPointCloud): nube de puntos como objeto de CloudCompare
    Returns:
        xyz(array): array numpy de la forma (x,y,z, scalarfields)
        scalarFiledsNames(list): nombre de campos escalares
    """
    #Init
    if not isinstance(cloud, pycc.ccPointCloud):
        raise RuntimeError("La nube de puntos de entrada debe ser una instance de pycc.ccPointCloud")

    #se pasa la nube de puntos a un array de numpy
    xyz = cloud.points() #ndarray(x,y,z)

    #comprobando la existencia de puntos
    if xyz.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

    #recuperando los valores de los campos escalares
    #key: nombre del campo escalar
    #value: valores del campo escalar
    scalar_fields_names = [cloud.getScalarFieldName(i) for i in range(cloud.getNumberOfScalarFields())]
    aux_dict = {}
    for idx, field_name in enumerate(scalar_fields_names):
        if field_name not in aux_dict.keys():
            aux_dict[field_name] = cloud.getScalarField(idx).asArray()

    #si la nube de puntos esta coloreada en RGBA, se recupera cada 'color' como un campo escalar
    if cloud.colors() is not None:
        aux_dict['Red'], aux_dict['Green'], aux_dict['Blue'], _ = cloud.colors().T

    #añadiendo todo al ndarray de salida
    for key, values in aux_dict.items():
        xyz = np.column_stack((xyz, np.array(values.astype(np.float64))))

    #scalarFieldsNmaes puede darse el caso en el que sea una lista vacia
    scalarFiledsNames = list(aux_dict.keys())
    return xyz, scalarFiledsNames



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
        point_cloud(ccPointCloud()): nube de puntos como objeto ccPointCloud()
    """
    #Init
    if cloud.shape[0]==0 or cloud.shape[1]<3:
        return

    #Instanciando un objeto pointcloud de CloudCompare
    x = cloud[:,0]
    y = cloud[:,1]
    z = cloud[:,2]
    point_cloud = pycc.ccPointCloud(x,y,z)
    point_cloud.setName(name)


    #Comprobación de la existencia de campo escalar Zcoord
    id = point_cloud.getScalarFieldIndexByName("Zcoord")
    if id == -1:
        point_cloud.addScalarField("Zcoord", z)
        point_cloud.setCurrentDisplayedScalarField(point_cloud.getScalarFieldIndexByName("Zcoord"))

    #añadiendo los demas campos escalares
    if cloud.shape[1]>3:
        if len(scalar_fields)>0:
            for idx, field_name in enumerate(scalar_fields):
                id = point_cloud.getScalarFieldIndexByName(field_name)
                if id == -1:
                    id = point_cloud.addScalarField(field_name, cloud[:,idx+3])
                    if id == -1:
                        raise RuntimeError("Failed to add ScalarField")

    #Calculo de valores mínimos y máximos de campos escalares
    sfs = [point_cloud.getScalarFieldName(i) for i in range(point_cloud.getNumberOfScalarFields())]
    for sf in sfs:
        point_cloud.getScalarField(point_cloud.getScalarFieldIndexByName(sf)).computeMinAndMax()

    return point_cloud


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
    #Init
    if len(clouds) ==0:
        raise RuntimeError("La lista de entrada esta vacía")

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
    #Init
    #comprobando la existencia de puntos
    if cloud.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

    # extracción de coordenadas
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
        Obtener nube de puntos por celda, la busqueda de vecinos se realiza mediante arbol cKDTree
    Args:
        xyz(ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda
    Returns:
        cells(array): lista de arrays, donde cada array contiene los cuatro puntos que definen una celda.
        clouds(list): lista de nubes de puntos por celda en formato ndarray de numpy
    """
    #Init
    #comprobando la existencia de puntos
    if xyz.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

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
        indices = tree.query_ball_point([(x_min + x_max) / 2, (y_min + y_max) / 2], d, workers=-1)

        #check
        if len(indices) == 0:
            continue

        #filtrando y añadiendo las nubes de puntos a la lista de salida
        cell_points = xyz[indices]
        clouds.append(cell_points)

    #Controlando la salida
    if len(clouds) > 0:
        return cells, clouds
    else:
        raise RuntimeError("La nube de puntos esta vacía")





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




#Análisis de componentes principales (PCA)
#@numba.njit
def customPCA(cloud, nPC):
    """
    _summary_:
        Aplica Análisis de Componentes Principales (PCA) a la nube de puntos de entrada
    Args:
        cloud (ndarray(x,y,z)): nube de puntos como array numpy
        nPC (int): número de componentes
    Returns:
        eigenvalues: los nPC primeros autovalores del análisis
        eigenvectors: los nPC primeros autovectores del análisis
    """
    #Init
    #comprobando la existencia de puntos
    if cloud.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

    #PCA
    pca = PCA(n_components=nPC)
    pca.fit(cloud)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    """
    m = np.mean(cloud, axis=0)
    cloud_norm = cloud - m
    cov = np.cov(cloud_norm.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov.T)
    sorted_indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:, sorted_indexes]
    return eigenvalues[:3], eigenvectors[:,:3].T
    """
    return eigenvalues, eigenvectors




#==============================================================================================================
#==================================Funciones especificas para Rail detection===================================
#==============================================================================================================

#Análisis de regresión lineal RANSAC
#@numba.njit No compatible con RANSACRegressor()
def customRansac(cloud):
    """
    _summary_:
        Regresion RANSAC para determinar puntos colineales en una nube de puntos con ruido
    Args:
        cloud (ndarray(x,y,z)): nube de puntos
    Returns:
        inliers(array): máscara de puntos que estan sobre la recta de regresión
        outliers(array): máscara de puntos que no estan sobre la recta de regresión
        m(float): pendiente de la recta de regresión
    """
    #Init
    #comprobando la existencia de puntos
    if cloud.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

    # Extraer las coordenadas x, y, z
    x = cloud[:, 0]
    y = cloud[:, 1]

    # Modelo de regresión RANSAC
    model_ransac = RANSACRegressor(min_samples=2,
                                    max_trials=100,
                                    loss='absolute_error',
                                    residual_threshold=0.03,
                                    stop_probability=0.99,
                                    random_state=123)
    #Ajuste del modelo RANSAC
    model_ransac.fit(x[:, np.newaxis], y)

    # Obtener los puntos que se ajustan al modelo (inliers y outliers)
    inliers = model_ransac.inlier_mask_
    outliers = np.logical_not(inliers)

    #pendiente de la recta de regresión
    m = model_ransac.estimator_.coef_[0]
    return inliers, outliers, m




#Obtener puntos potenciales de ser de carril en una nube de puntos
@numba.njit
def getTrackPoints(clouds):
    """
    _summary_:
        Filtrar los puntos de carril de vias ferreas presentes en una nube de puntos
    Args:
        clouds(list): lista con nubes de puntos de la forma ndarray(x,y,z,scalarfields)
    Returns:
        track_clouds(list): lista con nubes de puntos de la forma ndarray(x,y,z, scalarfields)
    """
    #Init
    #comprobando los datos de entrada
    if len(clouds) == 0 or not isinstance(clouds, list):
        raise RuntimeError("la entrada debe ser una lista de nube de puntos no vacía")

    # lista de salida
    track_clouds = []
    for cell_points in clouds:
        #Trabajamos con celdas que tengan más de 20 puntos (esto puede cambiarse)
        if cell_points.shape[0] > 20:
            #Percentil 10% de altura en cada celda
            mdt = getPercentil(cell_points[:, 2], 10)

            ##Condicion 1
            #Verificación de que menos del 10% de puntos se situen entre mdt+0.5 y mdt+4.5
            mask_1 = np.where((cell_points[:, 2] >= mdt+0.5) & (cell_points[:, 2] <= mdt+4.5))
            condition1 = (cell_points[mask_1].shape[0] / cell_points.shape[0])*100 #Porcentaje de puntos entre [mdt+0.5 , mdt+4.5]
            if condition1 < 10.0:
                #En las celdas que cumplen la condición, se filtran los puntos del terreno (z < mdt+0.5)
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
                    #estos puntos son potencialmente puntos de riel
                    condition3 = (track_points.shape[0] / ground_points.shape[0])*100
                    if condition3 < 50:
                        #nos quedamos con las celdas con más de 10 puntos
                        if track_points.shape[0] > 10:
                            track_clouds.append(track_points)
    return track_clouds








#==============================================================================================================
#========================================Funciones especificas para RPAS=======================================
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
        ground_point_clouds(ndarray): nube de puntos (ndarray) de terreno natural
        n_ground_point_clouds(ndarray): nubes de puntos (ndarray) de objetos sobre el terreno natural
    """
    #Init
    if len(clouds) == 0 or not isinstance(clouds, list):
        raise RuntimeError("la entrada debe ser una lista de nube de puntos no vacía")

    #algoritmo CSF
    csf = CSF.CSF()

    #parametros del algoritmo
    csf.params.bSloopSmooth = params[0]    # manejar pendientes pronunciadas después de la simulación
    csf.params.class_threshold = params[1] # umbral de clasificación puntos terreno y no terreno
    csf.params.cloth_resolution = params[2]# resolución de la tela
    csf.params.rigidness = params[3]       # 1)montaña y vegetacion densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    csf.params.interations = params[4]     # numero de iteraciones

    #filtramos por celdas
    f = clouds[0].shape[1]
    ground_point_clouds = np.empty((0,f))
    n_ground_point_clouds = np.empty((0,f))
    for cloud in clouds:
        #nubes de puntos con menos de 10 puntos las saltamos
        if cloud.shape[0] < 10:
            continue

        #Algoritmo Cloth Simulation Filter
        csf.setPointCloud(cloud)
        idx_ground = CSF.VecInt()             # una lista para indicar el índice de puntos de tierra después del cálculo
        idx_n_ground = CSF.VecInt()           # una lista para indicar el índice de puntos no terrestres después del cálculo
        csf.do_filtering(idx_ground, idx_n_ground)# filtrado

        #Indices de puntos terreno y no terreno
        idx_ground_array = np.array(idx_ground).astype(int)
        idx_n_ground_array = np.array(idx_n_ground).astype(int)

        #extracción de puntos terreno
        if idx_ground_array.shape[0] > 0:
            ground_points = cloud[idx_ground_array]
            ground_point_clouds = np.vstack((ground_point_clouds, ground_points))

        #extraccion de puntos sobre el terreno
        if idx_n_ground_array.shape[0] > 0:
            n_ground_points = cloud[idx_n_ground_array]
            n_ground_point_clouds = np.vstack((n_ground_point_clouds, n_ground_points))

    return np.unique(ground_point_clouds, axis=0), np.unique(n_ground_point_clouds, axis=0)





#Otención de indices espectrales de vegetación con los campos escalares RGB
#@numba.njit no compatible con diccionarios python
def getSpectralInfo(cloud, scalarfields):
    """
    _summary_:
        Obtención de indices espectrales de vegetación a partir de los campos escalares R,G,B
    Args:
        cloud (ndarray): nube de puntos como objeto ndarray de numpy
        scalarfields (list): lista python con el nombre de los campos escalares de la nube de puntos
    Returns:
        cloud(array): array numpy de la forma (x,y,z, scalarfields)
        scalarFieldsNames(list): nombre de campos escalares
    """
    #Init
    if cloud.shape[0] == 0:
        raise RuntimeError("la nube de puntos esta vacía")

    #Si no hay R,G,B. se devuelve la nube de puntos de entrada
    if ('Red' not in scalarfields) or ('Green' not in scalarfields) or ('Blue' not in scalarfields):
        print("No ha sido posible calcular índices espectrales. La nube de puntos no está coloreada en RGB.")
        return cloud, scalarfields

    #Obteniendo los valores de R,G,B de cada punto
    idx1 = scalarfields.index('Red') + 3
    idx2 = scalarfields.index('Green') + 3
    idx3 = scalarfields.index('Blue') + 3

    Red = cloud[:,idx1]
    Green = cloud[:,idx2]
    Blue = cloud[:,idx3]

    #========================================================================================
    #======================== Cálculo índices espectrales vegetación ========================
    #========================================================================================
    aux_dict = {}
    #Index Excess Green
    aux_dict['ExG'] = (2*Green - Red - Blue)

    #Index Excess Green Minus Excess Red
    aux_dict['EXG'] =(2*Green - Red - Blue) - (1.4*Red - Green)

    #Index Excess Red
    aux_dict['EXR'] = 1.4*Red - Green

    #Visible Normalized Difference Vegetation Index
    mask = np.any([Red==0, Green==0, Blue==0], axis=0)
    aux_dict['vNDVI'] = np.where(mask, 0, 0.5268*(Red**-0.1294 * Green**0.3389 * Blue**-0.31118))

    #brillo
    aux_dict['Brightness'] = 0.2126*Red + 0.7152*Green + 0.0722*Blue

    #Color Index of Vegetation
    aux_dict['CIVE'] = 0.441*Red - 0.881*Green + 0.385*Blue + 18.78745

    #Green Leaf Index
    numerador = 2*Green - Red - Blue
    denominador = 2*Green + Red + Blue
    aux_dict['GLI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Soil Adjusted Vegetation Index
    numerador = 1.5*(Green - Red)
    denominador = Green + Red + 0.5
    aux_dict['SAVI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Visual Atmospheric Resistance Index
    numerador = Green - Red
    denominador = Green + Red - Blue
    aux_dict['VARI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Green divide by Red
    aux_dict['GR'] = np.where(Red != 0, Green / Red, -9999)

    #Normalised Blue-Red Difference Index
    numerador = Red - Blue
    denominador = Red + Blue
    aux_dict['NBRDI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Normalised Green-Blue Difference Index
    numerador = Green - Blue
    denominador = Green + Blue
    aux_dict['NGBDI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Normalised Green-Red Difference Index
    numerador = Green - Red
    denominador = Green + Red
    aux_dict['NGRDI'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Normalized Greeness
    numerador = Green
    denominador = Green + Red + Blue
    aux_dict['NormG'] = np.where(denominador != 0, numerador / denominador, -9999)

    #Red Green Ratio Index
    aux_dict['RGRI'] = np.where(Green != 0, Red / Green, -9999)

    #========================================================================================
    #====================== Fin cálculo índices espectrales vegetación ======================
    #========================================================================================

    #Añadiendo todos los índices a la nube de puntos de entrada
    for key, values in aux_dict.items():
        cloud = np.column_stack((cloud, np.array(values.astype(np.float64))))

    #Añadiendo el nombre de los nuevos campos escalares de la nube de puntos
    NewScalarFields = list(aux_dict.keys())
    scalarFieldsNames = scalarfields + NewScalarFields

    #Eliminando valores -9999 en los indices GLI en adelante
    idx = scalarFieldsNames.index('GLI') + 3
    mask = np.all(cloud[:, idx:] != -9999, axis=1)
    cloud = cloud[mask]

    return cloud, scalarFieldsNames




#Otención de caracteristicas geométricas basadas en el vecindario de cada punto
#@numba.njit no compatible con cKDTree
def getGeometricalInfo(cloud, globalScalarfields, n):
    """
    _summary_:
        Obtención de caracteristicas geométricas basadas en el vecindario de cada punto de la nube de puntos
    Args:
        cloud (ndarray): nube de puntos como objeto ndarray de numpy
        globalScalarfields (list): lista python con el nombre de los campos escalares de la nube de puntos
        n(int): número de vecinos cercanos
    Returns:
        cloud_out(array): array numpy de la forma (x,y,z, scalarfields)
        scalarfields(list): nombre de campos escalares
    """

    #Init
    if cloud.shape[0] == 0:
        raise RuntimeError("la nube de puntos esta vacía")

    if cloud.shape[1] < 3:
        raise RuntimeError("la nube de puntos no es tridimensional")

    #función para el calculo de caracteristicas optimizada con numba
    @numba.njit()
    def compute(cloud, indices, scalarfields):

        #Lista para almacenar los puntos de salida
        cloud_out_list = numba.typed.List.empty_list(numba.float64[:])

        #iteramos la nube de puntos para obtener las nuevas caracteristicas a partir del vecindario de cada punto
        for idx in range(indices.shape[0]):

            #se obtiene el punto y su vecindario
            point = cloud[idx]
            vecindario = cloud[indices[idx]]

            #check del número de vecinos, si hay menos de 3 vecinos descartamos el punto, porque no se podrá aplicar PCA
            if vecindario.shape[0] < 3:
                continue

            #======================================================================
            #======================CALCULO DE CARACTERISTICAS======================
            #======================================================================

            #Z_std: desviación típica de las alturas de los puntos del vecindario
            Z_std = np.std(vecindario[:,2])

            ###calculo de caracteristcas de la descomposición en componentes principales###
            #descomposición en componentes principales y obtención de los 3 primeros autovalores
            coords = vecindario[:,:3]# coordenadas 3d
            cov = np.cov(coords.T) #matriz de covarianza
            eigenvalues, _ = np.linalg.eig(cov.T) #autovalores
            sorted_indexes = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indexes]
            cp1, cp2, cp3 = eigenvalues[:3]

            #Sum of eigenvalues
            sum_eigenvalues = cp1 + cp2 + cp3

            #Distribución tridimensional de los puntos de la vecindad
            omnivarianza = np.cbrt(cp1*cp2*cp3)

            #Entropía de Shannon de los valores propios
            ln = np.log(cp1)
            eigenentropy = -1*(cp1*ln + cp2*ln + cp3*ln)

            #Cambios en el vecindario en diferentes direcciones
            anisotropy = (cp1-cp3) / cp1

            #Bidimensionalidad de la vecindad en los ejes x e y
            planarity  = (cp2-cp3) / cp1

            #Dimensionalidad del vecindario en un eje
            linearity = (cp1-cp2) / cp1

            #Rugosidad superficial en las tres dimensiones
            surface_variation = cp3/sum_eigenvalues

            #Semejanza de la vecindad a la forma de una esfera
            sphericity = cp3/cp1

            #añadiendo las nuevas caracteristicas geométricas al array del punto
            attributes_geom = np.array([Z_std, sum_eigenvalues, omnivarianza, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity])
            point = np.concatenate((point, attributes_geom))

            #actualización de la lista de nombres de campos escalares
            new_names = ['Z_std', 'sum_eigenvalues', 'omnivarianza', 'eigenentropy', 'anisotropy', 'planarity', 'linearity', 'surface_variation', 'sphericity']
            names = [name for name in new_names if name not in scalarfields]
            scalarfields += names

            #calculo de la media y desviación típica del índice NGRDI
            if 'NGRDI' in scalarfields:
                #posición de los valores de índice NGRDI
                idx_NGRDI = scalarfields.index('NGRDI') + 3  # +3 porque las 3 primeras posiciones son las coordenadas x,y,z en la nube de puntos

                #calculo de características
                NGRDI_mean = np.mean(vecindario[:, idx_NGRDI])
                NGRDI_std = np.std(vecindario[:, idx_NGRDI])

                #añadiendo las nuevas características al array del punto
                attributes_geom = np.array([NGRDI_mean, NGRDI_std])
                point = np.concatenate((point, attributes_geom))

                #actualización de la lista de nombres de campos escalares
                new_names = ['NGRDI_mean', 'NGRDI_std']
                names = [name for name in new_names if name not in scalarfields]
                scalarfields += names

            #======================================================================
            #==================FIN CALCULO DE CARACTERISTICAS======================
            #======================================================================

            #añadiendo el punto a la lista de puntos de salida
            cloud_out_list.append(point)

        return cloud_out_list, scalarfields

    #Obtención de árbol cKD en el espacio 3D
        #leafsize: número de puntos por partición
        #compact_nodes: para reducir los hiperrectangulos al rango de datos real (consultas más rápidas)
        #balanced_tree: para usar en las particiones la mediana  y no la media (consultas más rápidas)
    tree = cKDTree(cloud[:,:3], leafsize=10, compact_nodes=True, balanced_tree=True)

    #índices de los 15 vecinos más próximos de cada punto
    _, indices_vecindarios = tree.query(cloud[:,:3], n, workers=-1)

    #calculo de caracteristicas
    cloud_out, newGlobalScalarfields = compute(cloud, indices_vecindarios, globalScalarfields)

    #conversion de lista numba a numpy array
    cloud_out = np.array(cloud_out)

    #si la lista de puntos de salida esta vacia, se retornan los datos iniciales
    if cloud_out.shape[0] == 0:
        return cloud, globalScalarfields

    # si no, se retorna la nube de puntos con las caraceristicas de vecindad de puntos
    else:
        return cloud_out, newGlobalScalarfields







#Clsificación no supervisada con GMM
#@numba.njit no compatible con RobustScaler y GaussianMixture
def clusteringGMM(cloud, globalScalarFieldsNames, cell_size):
    """
    _summary_:
        Esta función realiza una clasificación no supervisada de la nube de puntos en 3 clusters
        mediante el algotimo GMM (Gaussian Mixture Model), empleando para ello, índices de vegetación
        y características geométricas del vecindario de cada punto. La clasificación se realiza a nivel local
        en celdas de tamaño "cell_size". Si la nube de puntos no está coloreada en RGB, la clasificación se realiza
        mediante las caracteristicas geométricas, la intensidad de retorno y el número de retornos, con ello, los resultados
        pueden no ser adecuados.
    Args:
        cloud (ndarray): nube de puntos como objeto ndarray de numpy
        globalScalarFieldsNames (list): lista python con el nombre de los campos escalares de la nube de puntos
        cell_size(int, float): tamaño de ventana de análisis
    Returns:
        cloud_ground_gmm(array): array numpy de la forma (x,y,z, scalarfields)
        globalScalarFieldsNames(list): nombre de campos escalares
    """
    #Init
    if cloud.shape[0] == 0:
        raise RuntimeError("la nube de puntos esta vacía")

    if len(globalScalarFieldsNames) == 0:
        raise RuntimeError("la nube de puntos no tiene suficientes caracterisiticas para hacer clustering")

    #segmentamos a celdas de tamaño cell_size para hacer de forma local la clasificación
    _, cellPointCloud2 = getCellPointCloud(cloud, cell_size)

    #lista vacía para almacenar las nubes de puntos clasificadas
    clouds_ground_clas = []

    #caracteristicas definidas para la clasificación
    attributes_names = ['Intensity', 'Return Number', 'Red', 'Green', 'Blue', 'ExG', 'EXG', 'EXR', 'vNDVI', 'Brightness', 'CIVE', 'GLI',
                        'SAVI', 'VARI', 'GR', 'NBRDI', 'NGBDI', 'NGRDI', 'NormG', 'RGRI', 'NGRDI_mean', 'NGRDI_std', 'Z_std', 'sum_eigenvalues',
                        'omnivarianza', 'eigenentropy', 'anisotropy', 'planarity', 'linearity', 'surface_variation', 'sphericity']

    #se seleccionan, de la lista attributes_names, las caracteristicas existentes en los campos escalares de la nube de puntos
    existingAttributes = [name for name in  attributes_names if name in globalScalarFieldsNames]
    num_attributes = len(existingAttributes) #número de caracteristicas para la clasificación

    #comprobación de la existencia de caracteristicas para la clasificación
    if num_attributes == 0:
        raise RuntimeError("la nube de puntos no tiene suficientes caracterisiticas para hacer clustering")

    #Caracteristicas utilizadas para seleccionar el clutser de puntos terreno una vez hecha la clasificación
    #se usan estas variables, porque son donde el cluster de puntos de terreno toma un valor medio mínimo o máximo en comparación al resto de clusters
    minimos_in = [name for name in ['ExG', 'EXG', 'vNDVI', 'GLI', 'SAVI', 'VARI', 'GR', 'NBRDI', 'NGBDI', 'NormG', 'NGRDI_mean', 'NGRDI_std', 'Z_std', 'planarity', 'sphericity', 'omnivarianza', 'surface_variation'] if name in existingAttributes]
    maximos_in = [name for name in ['EXR', 'RGRI', 'CIVE', 'anisotropy'] if name in existingAttributes]
    names = minimos_in + maximos_in

    #comprobación de la existencia de caracteristicas para la selección de cluster terreno
    if len(names) == 0:
        raise RuntimeError("la nube de puntos no tiene las características utilizadas en la selección del cluster de puntos de terreno")

    #============================================================================================
    #==================================Proceso de clasificación==================================
    #============================================================================================
    #clasificamos a nivel de cada celda
    for cloud in cellPointCloud2:
    #Init
        #celdas con menos de 10 puntos las saltamos
        if cloud.shape[0] < 10:
            continue

        #comprobación de la NO existencia de valores nulos o infinitos en los datos para la clasificación
        #se descartan los puntos con valores nulos o infinitos en alguno de sus atributos
        #valores nulos
        if np.isnan(cloud).sum() > 0:
            indices_nulos = np.any(np.isnan(cloud), axis=1)
            cloud = cloud[~indices_nulos]
        #valores infinitos
        if np.isinf(cloud).sum() > 0:
            indices_inf = np.any(np.isinf(cloud), axis=1)
            cloud = cloud[~indices_inf]

        #Se vuelve a comprobar que la celda contenga más de 10 puntos
        if cloud.shape[0] < 10:
            continue

        #¡IMPORTANTE!:
        # Si la nube de puntos no esta coloreada en RGB, se utilizará para la clasificacion los campos escalares
        # Intensity, Return Number, Z_std, sum_eigenvalues, omnivarianza, eigenentropy, anisotropy, planarity, linearity,
        # surface_variation y sphericity. Con ello, el resultado puede no ser el adecuado.

    #A-)se crea un array que contendrá los datos para la clasificación (attributes_array)
        attributes_array = np.empty((cloud.shape[0], num_attributes))

        #se rellena el attributes_array con los valores de los campos escalares existingAttributes
        for col_index, name in enumerate(existingAttributes):
            idx = globalScalarFieldsNames.index(name) + 3  # +3 porque las 3 primeras posiciones son las coordenadas x,y,z en la nube de puntos inicial (Cloud)
            attributes_array[:, col_index] = cloud[:, idx]

    #B-)escalado de caracteristicas para un mejor funcionamiento del algoritmo de clasificación
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(attributes_array)

    #C-)clasificación - Gaussian Mixture Model Clustering
        #Covarianza tipo full: cada cluster tiene su propia matriz de covarianza general
        #n_cluster: número de grupos a formar
        n_cluster = 3
        gmm = GaussianMixture(n_components=n_cluster, covariance_type='full', random_state=42)
        gmm.fit(scaled_features)

        #probabilidades de pertenencia de cada punto a cada cluster
        probabilities = gmm.predict_proba(scaled_features)

        #probabilidad máxima y clase correspondiente para cada punto
        max_probabilities = np.max(probabilities, axis=1)
        labels_gmm = np.argmax(probabilities, axis=1)

        #añadimos, en la nube de puntos inicial, la clase predicha y la probabilidad de pertenencia a la clase
        cloud = np.column_stack((cloud, labels_gmm, max_probabilities))

        #============================================================================================
        #================================Fin proceso de clasificación================================
        #============================================================================================

    #D-)filtrado de puntos por probabilidades
        #definimos un umbral de probabilidad mínima para descartar puntos con una mala asignación a una clase
        #puntos con una probabilidad de pertenencia a un cluster menor que 60% se descarta
        mask = cloud[:,-1] >= 0.6
        cloud = cloud[mask]

        #============================================================================================
        #===========================Proceso de selección de cluster terreno==========================
        #============================================================================================
        #dic es una variable auxiliar que almacena:
            #key: cada variable existente en la lista names (names = minimos_in + maximos_in)
            #value: lista de valores medios de la distribución normal ajustada a los datos de cada key
        dic ={}

        #iteramos por cada cluster para obtener las métricas de comparación (mu: media de la distribución normal ajustada a los datos)
        for i in range(n_cluster):

            #obtención de cada cluster i=0,1,2 (en el caso de 3 clusters)
            labels_idx = -2 # en cloud, las labels estan en la penúltima posición
            mask = cloud[:,labels_idx] == i
            cluster = cloud[mask]

            #obtención del valor medio de la distribución normal ajustada a los datos de cada variable
            for name in names:
                #obtención del indice de cada variable de la lista names
                var_idx = globalScalarFieldsNames.index(name) + 3

                #calculo de estadisticos, para la selección del cluster de puntos terreno
                #solo se tiene en cuenta la media de la distribución (mu)
                #mu: media, sigma: desviación típica
                (mu, sigma) = stats.norm.fit(cluster[:, var_idx])

                #almacenando los resultados en la variable auxiliar dic
                if name not in dic.keys():
                    dic[name] = [mu]
                else:
                    dic[name].append(mu)

        #obtención de números de cluster candidatos a ser de terreno según cada variable
        #buscando el índice del valor mínimo o máximo según las variables en minimos_in y maximos_in
        min_indices = [values.index(min(values)) for key, values in dic.items() if key in minimos_in]
        max_indices = [values.index(max(values)) for key, values in dic.items() if key in maximos_in]

        #concatenación de las listas de número de clusters
        clusters = min_indices + max_indices

        #número de cluster seleccionado como terreno
        #se selecciona aquel número de cluster más frecuente entre los candidatos (moda de la lista de indices)
        clase = stats.mode(clusters).mode

        #filtrado de puntos por número de cluster
        mask = cloud[:,labels_idx] == clase
        cluster_ground = cloud[mask]

        #============================================================================================
        #=========================Fin proceso de selección de cluster terreno========================
        #============================================================================================

        #añadimos la nube de puntos a la lista de nubes de puntos clasificadas
        clouds_ground_clas.append(cluster_ground)

    #se actualiza la lista de nombres de los campos escalares
    globalScalarFieldsNames.append("Label_GMM")
    globalScalarFieldsNames.append("Probabilities_GMM")

    #uniendo los segmentos de nubes de puntos clasificados en un solo array de numpy
    cloud_ground_gmm = np.concatenate(clouds_ground_clas, axis=0)

    #eliminando puntos duplicados con base en las coordenadas x,y,z
    _, unique_indices = np.unique(cloud_ground_gmm[:, :3], axis=0, return_index=True)
    cloud_ground_gmm = cloud_ground_gmm[unique_indices]

    return cloud_ground_gmm, globalScalarFieldsNames
