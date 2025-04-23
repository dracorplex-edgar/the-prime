import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle

# Configuración
DATA_PATH = 'proyecto/DATA/pro_1'  
MODEL_DIR = 'proyecto/src/Prototipo_one/modelo'
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_gestos_landmarks.keras')  # Formato .keras
os.makedirs(MODEL_DIR, exist_ok=True)  # Crear directorio si no existe


gestos = os.listdir(os.path.join(DATA_PATH, 'Entrenamiento'))
NUM_CLASSES = len(gestos)

EPOCHS = 200
BATCH_SIZE = 16
INPUT_SHAPE = (63,)  # 21 landmarks * 3 coordenadas

def cargar_datos(ruta_base):
    """Carga y valida los datos de landmarks."""
    X, y = [], []
    
    for gesto in os.listdir(ruta_base):
        gesto_path = os.path.join(ruta_base, gesto)
        
        for archivo in os.listdir(gesto_path):
            if archivo.endswith('.npy'):
                data = np.load(os.path.join(gesto_path, archivo))
                if data.shape != INPUT_SHAPE:
                    raise ValueError(f"Archivo {archivo} tiene shape {data.shape}, esperado {INPUT_SHAPE}")
                X.append(data)
                y.append(gesto)  # Guardar nombre de clase
    
    return np.array(X), np.array(y)

# cargar y preparar datos
X_train, y_train_raw = cargar_datos(os.path.join(DATA_PATH, 'Entrenamiento'))
X_val, y_val_raw = cargar_datos(os.path.join(DATA_PATH, 'Validacion'))

# codificar etiquetas y guardar mapeo
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_val = le.transform(y_val_raw)

# Guardar mapeo de clases y LabelEncoder
class_indices = {i: gesto for i, gesto in enumerate(le.classes_)}
with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)


# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

# Calcular pesos de clases
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(np.argmax(y_train, axis=1)), 
                                   y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))

# Modelo , considerar agregar cosas 
def crear_modelo():
    model = Sequential([
        Dense(128, activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),# modificar y probar diferentes learning_rate valor predeterminado 0.001
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

model = crear_modelo()

# Callbacks 
callbacks = [
    EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs'))
]

# Entrenamiento con validación
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluación final
print("\nEvaluación en validación:")
results = model.evaluate(X_val, y_val, verbose=0)
print(f"  - Pérdida: {results[0]:.4f}")
print(f"  - Exactitud: {results[1]*100:.2f}%")
print(f"  - Precisión: {results[2]*100:.2f}%")
print(f"  - Recall: {results[3]*100:.2f}%")

#tensorboard --logdir logs  ejecutar para revision de datos por medio de interfaz de servidor de tensorflow
#en computadora con grafica un entrenamiento con 120 epocas se hace en 2 minutos y en laptop tardaba 30 minutos maximo