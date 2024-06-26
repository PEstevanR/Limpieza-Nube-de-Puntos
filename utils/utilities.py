
import CSF
import pycc
import cccorelib
import numba
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
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

    #si la nube de puntos esta colorizada en RGBA, se recupera cada 'color' como un campo escalar
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
        point_cloud(ccPointCloud()): nube de puntos resultado de la unión
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
        Obtener nube de puntos por celda, la busqueda de vecinos se realiza mediante arbol KDTree
    Args:
        xyz(ndarray(x,y,z)): nube de puntos
        cell_size (int, float): tamaño de celda
    Returns:
        cells(array): lista de arrays donde cada array contiene los cuatro puntos que definen una celda.
        clouds(list): lista de nubes de puntos por celda
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
        indices = tree.query_ball_point([(x_min + x_max) / 2, (y_min + y_max) / 2], d)

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






#==============================================================================================================
#================================================Rail detection================================================
#==============================================================================================================
#Análisis de componentes principales (PCA)
#@numba.njit
def customPCA(cloud, nPC):
    """
    _summary_:
        Aplica Análisis de Componentes Principales (PCA) a la nube de puntos
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
    return eigenvalues[:3], eigenvectors[:,:3].T"""
    return eigenvalues, eigenvectors


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
    #Init
    #comprobando la existencia de puntos
    if cloud.shape[0] == 0:
        raise RuntimeError("La nube de puntos esta vacía")

    # Extraer las coordenadas x, y, z
    x = cloud[:, 0]
    y = cloud[:, 1]

    # Aplicar la regresión RANSAC
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
    m = model_ransac.estimator_.coef_[0]
    return inliers, outliers, m




#Obtener puntos potenciales de ser de carril en una nube de puntos
@numba.njit
def getTrackPoints(clouds):
    """
    _summary_:
        Filtrar los puntos de carril presentes en una nube de puntos
    Args:
        clouds(list): lista con nubes de puntos ndarray(x,y,z)
    Returns:
        track_clouds(list): lista con nubes de puntos (ndarray(x,y,z) potencialmente de carril
    """
    #Init
    #comprobando la existencia de puntos
    if len(clouds) == 0 or not isinstance(clouds, list):
        raise RuntimeError("la entrada debe ser una lista de nube de puntos no vacía")

    # lista de salida
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





def getGeometricalInfo(cloud, scalarfields, d):
    """
    _summary_:
        Obtención de caracteristicas geométricas basadas en el vecindario de cada punto de la nube de puntos
    Args:
        cloud (ndarray): nube de puntos como objeto ndarray de numpy
        scalarfields (list): lista python con el nombre de los campos escalares de la nube de puntos
    Returns:
        cloud_out(array): array numpy de la forma (x,y,z, scalarfields)
        scalarfields(list): nombre de campos escalares
    """
    #Init
    if cloud.shape[0] == 0:
        raise RuntimeError("la nube de puntos esta vacía")

    if cloud.shape[1] < 3:
        raise RuntimeError("la nube de puntos no es tridimensional")

    #Arbol KD en el espacio 3D
    #leafsize: número de puntos por partición
    #compact_nodes: para reducir los hiperrectangulos al rango de datos real (consultas más rápidas)
    #balanced_tree: para usar en las particiones la mediana  y no la media (consultas más rápidas)
    tree = cKDTree(cloud[:,:3], leafsize=10, compact_nodes=True, balanced_tree=True)

    #Lista para almacenar los puntos de salida
    cloud_out_list = []

    #iteramos por cada punto en la nube de puntos para obtener las nuevas caracteristicas a partir de su vecindario
    for point in cloud:
        #coordenadas del punto
        x, y, z = point[:3]

        #vecinos del punto a una distancia d
        indices = tree.query_ball_point([x, y, z], d)
        vecindario = cloud[indices]

        #check del número de vecinos, si hay menos de 3 vecinos descartamos el punto (no se podrá aplicar PCA)
        if vecindario.shape[0] < 3:
            continue

        #======================================================================
        #======================CALCULO DE CARACTERISTICAS======================
        #======================================================================
        #número de vecinos
        n_vec = vecindario.shape[0]

        #dist_media: media de las distancias entre puntos del vecindario
        #dist_std: desviación típica de las distancias entre puntos del vecindario
        distances = np.linalg.norm(vecindario[:, :3] - point[:3], axis=1)
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)

        #Z_std: desviación típica de las alturas de los puntos del vecindario
        Z_std = np.std(vecindario[:,2])

        #calculo de caracteristcas de la descomposcion en componentes principales
        coords = vecindario[:,:3]
        #descomposición en componentes principales y obtención de los 3 primeros autovalores
        pca_3d = customPCA(coords, nPC=3)
        cp1, cp2, cp3 = pca_3d[0]

        #Sum of eigenvalues
        sum_eigenvalues = cp1 + cp2 + cp3 #suma de autovalores

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
        attributes_geom = [n_vec, dist_mean, dist_std, Z_std, sum_eigenvalues, omnivarianza, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity]
        point = np.concatenate((point, attributes_geom))

        #calculo de la media y desviación típica del índice NGRDI
        if 'NGRDI' in scalarfields:
            idx = scalarfields.index('NGRDI') + 3  # +3 porque las 3 primeras posiciones son las coordenadas x,y,z en la nube de puntos
            NGRDI_mean = np.mean(vecindario[:, idx])
            NGRDI_std = np.std(vecindario[:, idx])
            attributes_geom = [NGRDI_mean, NGRDI_std]
            point = np.concatenate((point, attributes_geom))
        #======================================================================
        #==================FIN CALCULO DE CARACTERISTICAS======================
        #======================================================================

        #añadiendo el punto a la lista de puntos de salida
        cloud_out_list.append(point)

    #si la lista de puntos de salida esta vacia, se retornan los datos iniciales
    if len(cloud_out_list) == 0:
        return cloud, scalarfields

    #convirtiendo la lista de puntos de salida en un array NumPy
    cloud_out = np.array(cloud_out_list)

    #añadiendo el nombre de los nuevos atributos a la lista de nombres
    attributes_geom_names = ['n_vec', 'dist_mean', 'dist_std', 'Z_std', 'sum_eigenvalues', 'omnivarianza', 'eigenentropy', 'anisotropy', 'planarity', 'linearity',
                             'surface_variation', 'sphericity']
    #si NGRDI_mean y NGRDI_std se han calculado, tambien se añaden en la lista de nombres de campos escalares
    if 'NGRDI' in scalarfields:
        attributes_geom_names += ['NGRDI_mean', 'NGRDI_std']

    #nombre de campos escalares de salida
    scalarfields += attributes_geom_names

    return cloud_out, scalarfields

