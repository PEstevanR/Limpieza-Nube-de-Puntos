#=============================================================================================
#=================================Script en desarrollo========================================
#=============================================================================================
"""
Objetivo:
        filtrar los puntos que son de terreno (ground points) para generar curvas de nivel

Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare

        2-)Reducción de puntos al 50% de forma aleatoria, para hacer más agil el procesamiento de datos

        3-)Reducción de ruido con el filtro 'filter_noise' de cloudcompare

        4-)Obtención de información espectral a partir de los campos escalares RGB

        5-)Filtrado espectral empleando los indices ExG y Brightness

        6-)Segmentación de la nube de puntos por celdas de paso 50m

        7-)Filtrado geométrico de puntos de terreno natural con el algoritmo Cloth Simulation Filter (CSF)

        8-)Carga de nubes de puntos a CloudCompare

Resultado:
"""

# Manejo de rutas de ejecución
import os
import sys

current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
#parent_dir="D:\LimpiezaNubes"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#modulos de cloudcompare
import pycc
import cccorelib

#otros
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

#Importación de funciones propias
from utils.utilities import getCloud, getCellPointCloud, mergePointClouds, makePyccPointCloudObject
from utils.utilities import filterCSF, getCloudAsArray, getSpectralInfo, getGeometricalInfo



#Aqui inicia la ejecución del código
if __name__=="__main__":

    print("Inicio de ejecución")

#1-)cargamos los datos desde CloudCompare
    cc, point_cloud, _ = getCloud()

#2-)reducción de puntos al 50%
    reduccion = point_cloud.size()//2
    subsample = cccorelib.CloudSamplingTools.subsampleCloudRandomly(cloud=point_cloud, newNumberOfPoints=reduccion)
    random_subsample_cloud = point_cloud.partialClone(subsample)
    #random_subsample.setName('subsample 50%')
    #cc.addToDB(random_subsample)


#3-)Eliminación de ruido
    #refcloud_2 = cccorelib.CloudSamplingTools.sorFilter(cloud=random_subsample, knn=6, nSigma=1.0)
    noisefilter = cccorelib.CloudSamplingTools.noiseFilter(cloud=random_subsample_cloud,
                                                          kernelRadius=0.1,
                                                          nSigma=1.0,
                                                          removeIsolatedPoints = True,
                                                          useKnn = False,
                                                          useAbsoluteError=False,
                                                          )
    denoised_cloud = random_subsample_cloud.partialClone(noisefilter)
    #denoised_cloud.setName('filtro de ruido')
    #cc.addToDB(denoised_cloud)


#4-)Conversión de la nube depuntos a ndarray numpy (x,y,z,globalScalarFields)
    cloudAsArray, globalScalarFieldsNames = getCloudAsArray(denoised_cloud)


#5-)Segmentación de la nube de puntos por celdas
    cell_size = 50
    cells, cellPointCloud = getCellPointCloud(cloudAsArray, cell_size)

    print("segmentacion hecha")

#6-)filtro (GEOMETRICO) de puntos terreno con Cloth Simulation Filter (CSF) en cada celda
    #parametros del algoritmo
    smooth = True     # manejar pendientes pronunciadas después de la simulación
    threshold = 0.5   # umbral de clasificación puntos terreno y no terreno
    resolution = 0.5  # resolución de la tela
    rigidness = 2     # 1)montaña y vegetacion densa, 2)escenas complejas, 3)terrenos planos con edificios altos
    interations = 500 # numero de iteraciones
    params = [smooth, threshold, resolution, rigidness, interations]
    ground_csf, n_ground_csf = filterCSF(cellPointCloud, params)

    print("filtrado geométrico hecho")


#7-)Clasificación no supervisada con base en índices espectrales de vegetación (algoritmo GMM - Gaussian Mixture Model)

    #se calculan índices espectrales
    cloud_espectralInfo, globalScalarFieldsNames = getSpectralInfo(cloud=ground_csf, scalarfields=globalScalarFieldsNames)

    #se calculan caracteristicas geométricas basadas en vecindarios
    cloud_geometricalInfo, globalScalarFieldsNames = getGeometricalInfo(cloud=cloud_espectralInfo, scalarfields=globalScalarFieldsNames, d=0.5)

    #segmentamos a celdas de 25m para evitar ruido en el clasificador GMM
    cell_size = 25
    _, cellPointCloud2 = getCellPointCloud(cloud_geometricalInfo, cell_size)

    #lista vacía para almacenar las nubes de puntos una vez clasificadas
    clouds_clas = []

    #clasificamos a nivel de cada celda
    for cloud in cellPointCloud2:

        #nubes de puntos con menos de 10 puntos las saltamos
        if cloud.shape[0] < 10 or cloud is None:
            continue

        #comprobación de la NO existencia de valores nulos o infinitos en los datos para la clasificación
        #valores nulos
        if np.isnan(cloud).sum() > 0:
            indices_nulos = np.any(np.isnan(cloud), axis=1)
            cloud = cloud[~indices_nulos]
        #valores infinitos
        if np.isinf(cloud).sum() > 0:
            indices_inf = np.any(np.isinf(cloud), axis=1)
            cloud = cloud[~indices_inf]

        #nubes de puntos con menos de 10 puntos las saltamos
        if cloud.shape[0] < 10:
            continue

        #==================================Proceso de clasificación==================================
        #A-)definción de las caracteristicas para la clasificación
            #Si la nube de puntos no esta coloreada en RGB, se utilizará para la clasificacion los campos escalares
            #Intensity, Return Number, n_vec, dist_mean', dist_std, Z_std, sum_eigenvalues, omnivarianza, eigenentropy,
            #anisotropy, planarity, linearity, surface_variation y sphericity
        attributes_names = ['Intensity', 'Return Number', 'Red', 'Green', 'Blue', 'ExG', 'EXG', 'EXR', 'vNDVI', 'Brightness', 'CIVE', 'GLI',
                            'SAVI', 'VARI', 'GR', 'NBRDI', 'NGBDI', 'NGRDI', 'NormG', 'RGRI', 'n_vec', 'NGRDI_mean', 'NGRDI_std', 'dist_mean',
                            'dist_std', 'Z_std', 'sum_eigenvalues', 'omnivarianza', 'eigenentropy', 'anisotropy', 'planarity', 'linearity',
                            'surface_variation', 'sphericity']
        existingAttributes = [name for name in  attributes_names if name in globalScalarFieldsNames]
        num_attributes = len(existingAttributes)

        #array vacio que contendrá la nube de puntos solo con los campos escalares para la clasificación
        attributes_array = np.empty((cloud.shape[0], num_attributes))

        #llena el attributes_array con los valores de los campos escalares attributes_names
        for col_index, name in enumerate(existingAttributes):
            idx = globalScalarFieldsNames.index(name) + 3  # +3 porque las 3 primeras posiciones son las coordenadas x,y,z en la nube de puntos inicial (Cloud)
            attributes_array[:, col_index] = cloud[:, idx]

        #B-)escalado de caracteristicas para un mejor funcionamiento del algoritmo de clasificación
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(attributes_array)

        #C-)clasificación
        #Gaussian Mixture Model Clustering
        #Covarianza tipo full: cada cluster tiene su propia matriz de covarianza general
        gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        gmm.fit(scaled_features)

        #obtener las probabilidades de pertenencia de cada punto a cada cluster
        probabilities = gmm.predict_proba(scaled_features)

        #determinar la probabilidad máxima y la clase correspondiente para cada punto
        max_probabilities = np.max(probabilities, axis=1)
        labels_gmm = np.argmax(probabilities, axis=1)

        #añadimos la clase predicha y la probabilidad de pertenencia a la clase en la nube de puntos inicial
        #añadimos los nombres de los campos escalares a la lista de nombres
        cloud = np.column_stack((cloud, labels_gmm, max_probabilities))

        #D-)filtrado de puntos por probabilidades y por número de cluster
        #definimos un umbral de probabilidad mínima para descartar puntos con una mala asignación a una clase
        mask = cloud[:,-1] >= 0.6 #p.e 60%
        cloud_ground_cluster = cloud[mask]

        ##############################################################
        ##########################PRUEBAS#############################
        if "Label_GMM" not in globalScalarFieldsNames and "Probabilities_GMM" not in globalScalarFieldsNames:
            globalScalarFieldsNames.append("Label_GMM")
            globalScalarFieldsNames.append("Probabilities_GMM")
        cloud = [makePyccPointCloudObject(cloud_ground_cluster, globalScalarFieldsNames, "Clustering")]
        merged_cloud= mergePointClouds(cloud, name_merge='Clustering')
        merged_cloud.setCurrentDisplayedScalarField(merged_cloud.getScalarFieldIndexByName("Zcoord"))
        cc.addToDB(merged_cloud)

        #añadimos la nube de puntos clasificada a la lista de nubes de puntos clasificadas
        #clouds_clas.append(cloud_ground_cluster)

    print("Clasificación hecha")
    """
    #actualizando la lista de nombres de los campos escalares
    #globalScalarFieldsNames = scalarFieldsNames
    globalScalarFieldsNames.append("Label_GMM")
    globalScalarFieldsNames.append("Probabilities_GMM")

    #uniendo los segmentos de nubes de puntos clasificados en un solo array de numpy
    #array numpy vacio con el mismo número de columnas que el de las nubes de puntos clasificadas
    f = clouds_clas[0].shape[1]
    ground_gmm = np.empty((0,f))

    #iterando sobre la lista de nubes de puntos clasificadas
    for cloud in clouds_clas:
        ground_gmm = np.vstack((ground_gmm, cloud))

    #eliminando puntos duplicados con base en las coordenadas x,y,z
    _, unique_indices = np.unique(ground_gmm[:, :3], axis=0, return_index=True)
    ground_gmm = ground_gmm[unique_indices]


    cloud = [makePyccPointCloudObject(ground_gmm, globalScalarFieldsNames, "Clustering")]
    merged_cloud= mergePointClouds(cloud, name_merge='Clustering')
    merged_cloud.setCurrentDisplayedScalarField(merged_cloud.getScalarFieldIndexByName("Zcoord"))
    cc.addToDB(merged_cloud)





    
#8-)Generación de malla 3D

    a = pycc.ccMesh()
    cccorelib.Delaunay2dMesh.buildMesh()
    a.#(cccorelib.TRIANGULATION_TYPES.DELAUNAY_2D_AXIS_ALIGNED, dim=2)

    mesh1 = cc.ccMesh.triangulate(cloud1, cc.TRIANGULATION_TYPES.DELAUNAY_2D_AXIS_ALIGNED, dim=2)
    mesh1.setName("mesh1")
    """

#9-)Carga de nubes de puntos a CloudCompare
    cc_ground_clouds = [makePyccPointCloudObject(ground_csf, scalarFieldsNames, "Ground points")]
    cc_n_ground_clouds = [makePyccPointCloudObject(n_ground_csf, scalarFieldsNames, "Non Ground points")]

    #union de nubes de puntos
    #ground points
    merged_ground_points= mergePointClouds(cc_ground_clouds, name_merge='Ground points')
    merged_ground_points.setCurrentDisplayedScalarField(merged_ground_points.getScalarFieldIndexByName("Zcoord"))

    #no ground points
    merged_n_ground_points= mergePointClouds(cc_n_ground_clouds, name_merge='Non Ground points')
    merged_n_ground_points.setCurrentDisplayedScalarField(merged_n_ground_points.getScalarFieldIndexByName("Zcoord"))

    #resultado a cloudcompare
    cc.addToDB(merged_ground_points)
    cc.addToDB(merged_n_ground_points)
    cc.updateUI()

    """
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
    """


    #del cc, point_cloud, random_subsample_cloud, denoised_cloud, xyz, ground_csf, n_ground_csf,


    print("PROCESO FINALIZADO")














    """
#5-)filtro (ESPECTRAL) basado en los campos escalares Brightness y ExG
    if 'Brightness' in scalar_fields and 'ExG' in scalar_fields:
        #umbrales
        umbral_ExG = 25
        umbral_Brightness = [50,150]
        idx1 = list(scalar_fields).index('Brightness')+3
        idx2 = list(scalar_fields).index('ExG')+3

        #filtro 1: Brightness <= 50 ó Brightness>=150 (sombras y terreno)
        mask_ground = np.logical_or(xyz[:,idx1] <= umbral_Brightness[0], xyz[:,idx1] >= umbral_Brightness[1])
        mask_n_ground = np.logical_not(mask_ground)
        ground_points_1 = xyz[mask_ground]
        n_ground_points_1 = xyz[mask_n_ground]

        #filtro 2: ExG <= 25 (terreno)
        mask_ground = ground_points_1[:,idx2] <= umbral_ExG
        mask_n_ground = np.logical_not(mask_ground)
        n_ground_points_2 = ground_points_1[mask_n_ground]

        #Resultado
        ground_points = ground_points_1[mask_ground]
        n_ground_points =  np.vstack((n_ground_points_1, n_ground_points_2))

    print("filtrado espectral hecho")
    """