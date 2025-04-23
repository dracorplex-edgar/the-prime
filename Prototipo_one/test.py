#visualizar archivos .nyp donde se guardan los datos del prototipo 1
import numpy as np

# Cargar el archivo
data = np.load('proyecto/DATA/pro_1/Entrenamiento/A_Gesto/Mano_0.npy')

# Mostrar el contenido
print(data)

print(data[:3],"coordenadas del punto de referencia seria el punto 0 que seria la muneca  ") 
#dimesiones 
print("Shape (dimensiones):", data.shape)
print("Numero de dimensiones:", data.ndim)
print("Tamaño total (elementos):", data.size)