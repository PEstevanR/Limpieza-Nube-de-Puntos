@echo off
set PYTHON_EXECUTABLE="C:\Program Files\CloudCompare\plugins\Python\python.exe"



echo .
echo "===================================================="
echo "==================Interprete python================="
echo "===================================================="
echo .
echo %PYTHON_EXECUTABLE%
echo .



:: Obtener la ruta al directorio de instalación de paquetes python
set DIR_PACKAGE="Lib\site-packages"
for %%A in (%PYTHON_EXECUTABLE%) do (
    set TARGET_DIR=%%~dpA
)
set TARGET_DIR=%TARGET_DIR%%DIR_PACKAGE%
set "TARGET_DIR=%TARGET_DIR:"=%"
set TARGET_DIR="%TARGET_DIR%"

echo .
echo "===================================================="
echo "========Directorio de instalacion de paquetes======="
echo "===================================================="
echo .
echo %TARGET_DIR%
echo .


rem Instalar paquetes
::%PYTHON_EXECUTABLE% -m pip install --upgrade pip
echo .
echo "===================================================="
echo "==================Instalando numba=================="
echo "===================================================="
echo .
%PYTHON_EXECUTABLE% -m pip install numba==0.60.0 --target=%TARGET_DIR%
echo .




echo .
echo "===================================================="
echo "=========Instalando cloth-simulation-filter========="
echo "===================================================="
echo .
%PYTHON_EXECUTABLE% -m pip install cloth-simulation-filter==1.1.5 --target=%TARGET_DIR%
echo.



echo.
echo "===================================================="
echo "=====Fin del proceso de instalacion de paquetes====="
echo "===================================================="
echo .
pause

