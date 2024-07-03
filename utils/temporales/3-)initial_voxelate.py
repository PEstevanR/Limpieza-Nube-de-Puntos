

import pycc
import dendromatics as dm

"""
Objetivo:
        Obtener una nube de puntos voxelizada a un tamaño de voxel en x, y, z determinado
Proceso:
        1-)Se carga la nube de puntos seleccionada en CloudCompare y se realiza una copia para que sea writeable
        2-)Se voxeliza la nube de puntos original empleando la utilidad voxelate del paquete dendromatics
        3-)Se añade la nube de puntos voxelizada a cloudcompare
Resultado:
        Se consigue voxelizar y renderizar la nube de puntos en la GUI de cloudCompare.
        Se obtiene nube de puntos homogénia
"""

def main():
    #Para interactuar con los datos seleccionados en CloudCompare, devuelve un objeto pycc.ccPythonInstance()
    cc = pycc.GetInstance()
#1-)
#de CloudCompare a objeto ccPointCloud
    entities = cc.getSelectedEntities() #Lista de objetos ccPointCloud
    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    point_cloud = entities[0] # primer objeto ccPointCloud
    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    #de objeto ccPointCloud a ndarray(x y z) numpy
    xyz_CloudCompare = point_cloud.points()

    #copia de la nube de puntos original
    xyz = xyz_CloudCompare.copy()
    xyz.flags.writeable = True #ndarray writeable


#2-)
#Voxelizacion
    #tamaño de voxel
    res=0.25
    voxelated_cloud, vox_to_cloud_ind, cloud_to_vox_ind = dm.voxelate(cloud=xyz, resolution_xy=res, resolution_z=res, with_n_points=False)

    x = voxelated_cloud[:,0]
    y = voxelated_cloud[:,1]
    z = voxelated_cloud[:,2]

#3-)
#Nube de puntos voxelizada
    voxels = pycc.ccPointCloud(x,y,z)
    voxels.setName("voxelated_cloud")
    #scalarfield coordenada Z (solo para visualizar)
    voxels.addScalarField("Z", z)
    voxels.setCurrentDisplayedScalarField(voxels.getScalarFieldIndexByName("Z"))
    cc.addToDB(voxels)

#Nube de puntos original por indices de vox
    cloud = xyz[vox_to_cloud_ind]
    x1 = cloud[:,0]
    y1 = cloud[:,1]
    z1 = cloud[:,2]
    voxels_to_point_cloud = pycc.ccPointCloud(x1,y1,z1)
    voxels_to_point_cloud.setName("xyz[vox_to_cloud_ind]")
    voxels_to_point_cloud.addScalarField("Z", z1)
    voxels_to_point_cloud.setCurrentDisplayedScalarField(voxels_to_point_cloud.getScalarFieldIndexByName("Z"))
    cc.addToDB(voxels_to_point_cloud)


#Nube de puntos vox to cloud
    cloud = voxelated_cloud[vox_to_cloud_ind]
    x1 = cloud[:,0]
    y1 = cloud[:,1]
    z1 = cloud[:,2]
    voxels_to_point_cloud = pycc.ccPointCloud(x1,y1,z1)
    voxels_to_point_cloud.setName("voxelated_cloud[vox_to_cloud_ind]")
    voxels_to_point_cloud.addScalarField("Z", z1)
    voxels_to_point_cloud.setCurrentDisplayedScalarField(voxels_to_point_cloud.getScalarFieldIndexByName("Z"))
    cc.addToDB(voxels_to_point_cloud)

    cc.updateUI()

if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO")