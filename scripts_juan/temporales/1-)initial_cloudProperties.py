

import pycc

"""
Objetivo:
        Entender la interaccion de pycc con CloudCompare.

Proceso:
        1-)Cargar la nube de puntos seleccionada en CloudCompare
        2-)Mostrar propiedades de la nube de puntos
        3-)Interactuar con la GUI de CloudCompare

Resultado:
        se consigue interactuar con las propiedades de la nube de puntos y la GUI de CloudCompare
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
#2-)
#Algunas propiedades y metodos de los objetos ccPointCloud

    print('El nombre de la nube es:', point_cloud.getName())
    print("Los colores de cada punto son : ", point_cloud.colors()) #RGB?
    print("Las coordenadas de cada punto son: ", point_cloud.points()) #XYZ
    print("Numero de puntos [n.ptos y n.coords]: ", point_cloud.points().shape)
    print("Valores del ScalarField por nombre: ", point_cloud.getScalarField(0).asArray()) #valores del scalar field 0


#Campos ecalares

    #El número de campos escalares se refiere a:
    #    en ficheros .laz: ['Intensity', 'Return Number', 'Number Of Returns', 'EdgeOfFlightLine', 'Classification', 'Point Source ID', 'Gps Time']

    print("Numero de campos escalares: ", point_cloud.getNumberOfScalarFields())
    print('Indice de determinado campo escalar (p.e, Intensity)', point_cloud.getScalarFieldIndexByName('Intensity'))

    scalar_fields: list[str] = []
    for i in range(point_cloud.getNumberOfScalarFields()):
        scalar_fields.append(point_cloud.getScalarFieldName(i))
    print("los nombres de los campos escalares son: ", scalar_fields)

#3-)
    #Congelar o descongelar la GUI
    cc.freezeUI(True)
    cc.freezeUI(False)

    #Desactivar visualizacion de la nube de puntos
    point_cloud.setEnabled(False)

    #Actualizar la sesion de cloudcompare
    cc.updateUI()



if __name__=="__main__":
    main()
    print("PROCESO FINALIZADO")


