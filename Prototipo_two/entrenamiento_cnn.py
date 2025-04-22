import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.models import Sequential, Model
import os
import matplotlib.pyplot as plt

# --- Configuración ---
# Rutas a tus carpetas de datos
direccion_base = 'proyecto/DATA/pro_1'
direccion_entrenamiento = os.path.join(direccion_base, 'Entrenamiento')
direccion_validacion = os.path.join(direccion_base, 'Validacion')

# Tamaño de las imágenes para la entrada del modelo (MobileNetV2 espera 224x224 por defecto)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3 # MobileNetV2 espera 3 canales (color)

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 20 # Puedes ajustar esto. Con early stopping, puede parar antes.
LEARNING_RATE = 0.0001 # Tasa de aprendizaje inicial (puede ser más baja para transfer learning)

# Ruta para guardar el modelo entrenado
modelo_guardar_ruta = 'modelo_lenguaje_senas.keras' # Nuevo formato recomendado .keras

# --- Cargar Datos ---
print("Cargando datos de entrenamiento...")
# image_dataset_from_directory infiere las clases de los nombres de las carpetas
train_ds = image_dataset_from_directory(
    directory=direccion_entrenamiento,
    labels='inferred',
    label_mode='categorical', # Para clasificación multi-clase con softmax
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest', # Método de redimensionamiento
    batch_size=BATCH_SIZE,
    shuffle=True # Mezclar datos de entrenamiento
)

print("Cargando datos de validación...")
val_ds = image_dataset_from_directory(
    directory=direccion_validacion,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False # No es necesario mezclar datos de validación
)

# Obtener el nombre de las clases (nombres de las carpetas) y el número de clases
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Clases detectadas: {class_names}")
print(f"Número de clases: {num_classes}")

# --- Preparación de Datos y Aumento de Datos ---

# MobileNetV2 espera que los píxeles estén en el rango [-1, 1].
# keras.applications.mobilenet_v2.preprocess_input hace esta normalización.
# Sin embargo, image_dataset_from_directory ya escala a [0, 255] por defecto.
# Podemos agregar una capa de rescaling o usar la función preprocess_input
# directamente después del aumento de datos.
# Una capa de Reescalado a [0, 1] es un buen primer paso general:
rescale = keras.layers.Rescaling(1./255)

# Capas de Aumento de Datos (elige las apropiadas para gestos)
# ¡Precaución con RandomFlip! Si un gesto es el reflejo de otro (ej: B y D en ASL),
# NO uses RandomFlip('horizontal'). Para señas generales, puede ser útil.
data_augmentation = Sequential([
  RandomRotation(factor=0.1),      # Rotación aleatoria ligera
  RandomZoom(height_factor=0.1, width_factor=0.1), # Zoom aleatorio ligero
  RandomTranslation(height_factor=0.1, width_factor=0.1), # Traslación aleatoria ligera
  # RandomFlip("horizontal"), # Descomenta con precaución si es aplicable
], name="data_augmentation")


# Función para preprocesar los datos (reescalado + aumento + normalización para MobileNetV2)
def preprocess(image, label):
    image = rescale(image) # Escalar a [0, 1]
    image = data_augmentation(image) # Aplicar aumento solo al entrenamiento
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Asegurar tipo float32
    # Nota: MobileNetV2 preprocess_input espera un rango diferente.
    # Si usas MobileNetV2 preprocess_input, no uses rescale a 1./255 antes.
    # Usaremos el flujo más común: [0, 255] -> [0, 1] -> Aumento -> feed to model
    # Las capas internas de MobileNetV2 manejan la normalización necesaria después.
    return image, label

# Mapear la función de preprocesamiento a los datasets
# Aplicar aumento solo al conjunto de entrenamiento
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(lambda x, y: (rescale(x), y)) # Solo reescalar validación


# Optimizar rendimiento con caching y prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# --- Construir el Modelo con Transfer Learning ---

# Cargar el modelo base MobileNetV2 pre-entrenado en ImageNet
# include_top=False quita la capa de clasificación final
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                         include_top=False,
                         weights='imagenet')

# Congelar el modelo base para que sus pesos no se entrenen inicialmente
base_model.trainable = False

# Crear el modelo completo
# Usamos la API Funcional de Keras para más flexibilidad
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# No necesitamos la capa de aumento aquí si la aplicamos al dataset arriba
# x = data_augmentation(inputs)
# x = keras.applications.mobilenet_v2.preprocess_input(x) # Si usas esta normalización

# Pasar la entrada (ya preprocesada en el dataset) a través del modelo base
x = base_model(inputs, training=False) # training=False asegura que las capas como BatchNormalization
                                       # en el modelo base corran en modo inferencia (usando estadísticas calculadas)

# Agregar capas de clasificación adicionales encima del modelo base
x = GlobalAveragePooling2D()(x) # Reducir las dimensiones espaciales
x = Dropout(0.2)(x) # Capa de Dropout para regularización
prediction_layer = Dense(num_classes, activation='softmax') # Capa de salida

outputs = prediction_layer(x)

model = Model(inputs=inputs, outputs=outputs)

# --- Compilar el Modelo ---

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar el resumen del modelo
model.summary()


# --- Callbacks (Para controlar el entrenamiento) ---

# Callback para guardar el mejor modelo basado en la precisión de validación
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=modelo_guardar_ruta,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', # Queremos maximizar la precisión de validación
    verbose=1
)

# Callback para detener el entrenamiento si la precisión de validación no mejora
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5, # Número de épocas sin mejora para detener
    mode='max',
    verbose=1,
    restore_best_weights=True # Restaurar los pesos del mejor epoch
)

# Lista de callbacks a usar
callbacks_list = [checkpoint_callback, early_stopping_callback]


# --- Entrenar el Modelo ---
print("Comenzando entrenamiento...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list
)

print("Entrenamiento finalizado.")
print(f"El mejor modelo se ha guardado en: {modelo_guardar_ruta}")


# --- Opcional: Visualizar el historial de entrenamiento ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()

# --- Evaluación Final ---
print("\nEvaluando el mejor modelo en el conjunto de validación:")
best_model = keras.models.load_model(modelo_guardar_ruta) # Cargar el mejor modelo guardado
loss, accuracy = best_model.evaluate(val_ds)
print(f"Pérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f}")
