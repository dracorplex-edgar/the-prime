import sys
import cv2
import os
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal # Import pyqtSignal for potential future use

# Desactivar GPU si no se necesita
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WRIST = 0
MIDDLE_FINGER_MCP = 9 # Metacarpophalangeal joint of the middle finger

# Asegúrate de que esta ruta sea correcta desde donde ejecutas el script
BASE_DIR = 'proyecto/src/Prototipo_one/modelo'
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_gestos_landmarks.keras')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')


def preprocess_landmarks(hand_landmarks):

    if not hand_landmarks or not hand_landmarks.landmark:
        # Devuelve un vector de ceros si no hay landmarks válidos
        return np.zeros(21 * 3)
    landmarks_list = hand_landmarks.landmark # lista de landmarks
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])

    reference_point = landmarks_array[WRIST].copy()
    landmarks_relative = landmarks_array - reference_point

    middle_finger_mcp_coords = landmarks_relative[MIDDLE_FINGER_MCP]
    scale_factor = np.linalg.norm(middle_finger_mcp_coords)

    if scale_factor < 1e-6:
        return np.zeros(21 * 3)

    landmarks_normalized = landmarks_relative / scale_factor

    return landmarks_normalized.flatten()



# Configuración condicional de Qt
if sys.platform.startswith('linux'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    # Solo establecer el path si existe realmente
    qt_plugin_path = '/usr/lib/x86_64-linux-gnu/qt5/plugins'  # Ruta común en Ubuntu/Debian
    if os.path.exists(qt_plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

class CameraApp(QWidget):
    # Opcional: Señal para emitir el gesto detectado si quieres usarlo en otro lado
    gesture_detected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección de Gestos en PyQt")
        self.setGeometry(100, 100, 800, 700) # Aumentar un poco altura para texto
        self.last_predicted_class = None  # Atributo de clase
        self.start_time = None
        self.current_duration = 0
        
        self.model = None
        self.class_map = None
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.prediction_text = "Inicializando..."

        try:
            if not os.path.exists(MODEL_PATH):
                 raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")
            if not os.path.exists(CLASS_INDICES_PATH):
                 raise FileNotFoundError(f"No se encontró el archivo de índices de clase en: {CLASS_INDICES_PATH}")

            print("Cargando modelo...")
            self.model = load_model(MODEL_PATH)
            print("Modelo cargado.")
            print("Cargando índices de clases...")
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            # Convertir claves de string a int para el mapeo
            self.class_map = {int(k): v for k, v in class_indices.items()}
            print("Índices cargados.")

        except FileNotFoundError as e:
            print(f"Error de archivo: {e}")
            self.prediction_text = f"Error al cargar: {e}"
            # sys.exit(1)
        except Exception as e:
            print(f"Error inesperado al cargar modelo/índices: {type(e).__name__} - {e}")
            self.prediction_text = "Error al inicializar modelo."

        try:
            print("Configurando MediaPipe Hands...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,      # Procesar video, no imágenes estáticas
                max_num_hands=1,              # Detectar solo una mano
                min_detection_confidence=0.7, # Umbral de detección inicial
                min_tracking_confidence=0.5   # Umbral para seguir la mano una vez detectada
            )
            self.mp_drawing = mp.solutions.drawing_utils # Para dibujar landmarks
            print("MediaPipe configurado.")
        except Exception as e:
             print(f"Error inesperado al configurar MediaPipe: {type(e).__name__} - {e}")
             self.prediction_text = "Error al inicializar MediaPipe."
             # sys.exit(1)

        print("Configurando cámara...")
        if sys.platform == 'win32':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            self.prediction_text = "Error: No se pudo abrir la cámara"
            # Aquí podrías deshabilitar el timer o mostrar un mensaje permanente
            # sys.exit(1) # Salir si la cámara es esencial
        else:
            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Cámara abierta ({self.cam_width}x{self.cam_height})")

        self.init_ui()

        self.timer = QTimer(self) # Pasar self como padre
        self.timer.timeout.connect(self.update_frame)
        if self.cap.isOpened(): # Solo iniciar timer si la cámara funciona
             self.timer.start(30) # 33 FPS pongamoslo asi 

    def init_ui(self):
        print("Inicializando UI...")
        layout = QVBoxLayout(self) # Pasar self para que el layout pertenezca al widget

        self.camera_label = QLabel(self) # Padre self
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) # Para escalar mejor
        # self.camera_label.setText("Esperando cámara...")
        layout.addWidget(self.camera_label) # No necesita stretch si es el único widget principal

        # Etiqueta para mostrar el texto de predicción (alternativa/adicional al texto en imagen)
        self.prediction_label = QLabel(self.prediction_text, self) # Padre self
        self.prediction_label.setAlignment(Qt.AlignCenter)
        font = self.prediction_label.font()
        font.setPointSize(14)
        self.prediction_label.setFont(font)
        self.prediction_label.setMaximumHeight(40) # Altura fija
        layout.addWidget(self.prediction_label)

        # Cuadro de texto (si aún lo necesitas)
        self.text_edit = QTextEdit(self) # Padre self
        self.text_edit.setPlaceholderText("Área de texto (opcional)")
        self.text_edit.setMaximumHeight(100)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)
        print("UI inicializada.")


    def update_frame(self):
        if not self.cap.isOpened():
            return # No hacer nada si la cámara no está lista

        ret, frame = self.cap.read()
        if not ret:
            print("Advertencia: No se pudo leer el frame de la cámara.")
            return

        # 1. Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Optimización

        # 2. Procesar con MediaPipe
        results = self.hands.process(frame_rgb)

        frame_rgb.flags.writeable = True # Permitir escritura de nuevo si fuera necesario

        # 3. Dibujar y Predecir (sobre el frame BGR original 'frame')
        current_prediction_text = "Gesto no entendeshion" #texto arriba del gesto

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks y conexiones en el frame BGR
                #self.mp_drawing.draw_landmarks(
                #    frame,                  # Imagen BGR donde dibujar
                #    hand_landmarks,         # Landmarks a dibujar
                #   self.mp_hands.HAND_CONNECTIONS # lineas o conecciones
                #)

                # Normalizar los landmarks para el modelo
                landmarks_vector = preprocess_landmarks(hand_landmarks)

                # Asegurarse de que la normalización produjo un vector válido
                if np.any(landmarks_vector) and self.model and self.class_map:
                    # Reformatear para que tenga la forma (1, 63) que espera el modelo
                    landmarks_vector_reshaped = landmarks_vector.reshape(1, -1)

                    # Realizar la predicción
                    try:
                        prediction = self.model.predict(landmarks_vector_reshaped, verbose=0)
                        predicted_class_index = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])

                        # Aplicar umbral de confianza y se puede ajustar 
                        if confidence > 0.80: 
                            
                            predicted_class_name = self.class_map.get(predicted_class_index, "Clase Desconocida")
                            current_prediction_text = f"{predicted_class_name} ({confidence*100:.1f}%)"
                            
                            if current_prediction_text != self.last_predicted_class:
                                self.start_time = time.time()
                                self.last_predicted_class = current_prediction_text
                            else:
                                self.current_duration = time.time() - self.start_time
                                if self.current_duration >= 1.5:  # segundos
                                    palabra = current_prediction_text[0].upper()
                                    current_text = self.text_edit.toPlainText()
                                    self.text_edit.setPlainText(current_text + palabra)
                                    self.start_time = time.time()
                                                        
                        else:
                            current_prediction_text = "Gesto no reconocido"
                    except Exception as e:
                         print(f"Error durante la predicción: {e}")
                         current_prediction_text = "Error en predicción"

                else:
                    # Si la normalización falla o el modelo no está listo
                    current_prediction_text = "Error en normalización"
        # else:
             # No se detectaron manos, current_prediction_text ya está puesto

        # etiqueta de texto abajo del frame
        self.prediction_label.setText(current_prediction_text)

        # Poner el texto de predicción directamente en la imagen 
        #cv2.putText(
        #    frame,                          # Imagen BGR
        #    current_prediction_text,
        #    (10, 30),                       # Posición (x, y) desde arriba-izquierda
        #    cv2.FONT_HERSHEY_SIMPLEX,       # Fuente
        #    1,                              # Tamaño de fuente
        #    (0, 255, 0),                    # Color (Verde en BGR)
        #    2,                              # Grosor
        #    cv2.LINE_AA                     # Mejor calidad de línea
        #)

        
        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = display_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        
        pixmap = QPixmap.fromImage(qt_image)
        # Escalar al tamaño actual del QLabel
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)


    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.hands:
            self.hands.close()
        print("Recursos liberados.")
        event.accept() 

if __name__ == "__main__":
    # Validar existencia de directorio base antes de iniciar la app
    if not os.path.isdir(BASE_DIR):
        print(f"Error Crítico: El directorio base '{BASE_DIR}' no existe.")
        print("Asegúrate de que la ruta sea correcta y el directorio contenga el modelo y los índices.")
        sys.exit(1) # Salir si el directorio base no existe

    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())