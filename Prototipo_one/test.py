import tensorflow as tf
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
print("Versión de CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
print("Versión de cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])