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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import PyQt5


# Desactivar GPU para TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def configure_qt():
    if sys.platform.startswith('linux'):
        # Usar los plugins que vienen con PyQt5
        qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
        if os.path.exists(qt_plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        return True
    return True


WRIST = 0
MIDDLE_FINGER_MCP = 9

BASE_DIR = os.path.join('proyecto', 'src', 'Prototipo_one', 'modelo')
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_gestos_landmarks.keras')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')

def preprocess_landmarks(hand_landmarks):
    if not hand_landmarks or not hand_landmarks.landmark:
        return np.zeros(21 * 3)
    
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    reference_point = landmarks_array[WRIST].copy()
    landmarks_relative = landmarks_array - reference_point
    scale_factor = np.linalg.norm(landmarks_relative[MIDDLE_FINGER_MCP])
    
    return (landmarks_relative / scale_factor).flatten() if scale_factor >= 1e-6 else np.zeros(21 * 3)

class CameraApp(QWidget):
    gesture_detected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección de Gestos en PyQt")
        self.setGeometry(100, 100, 800, 700)
        
        self.last_predicted_class = None
        self.start_time = None
        self.current_duration = 0
        
        self.init_models()
        self.init_camera()
        self.init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        if self.cap.isOpened():
            self.timer.start(30)  # ~33 FPS

    def init_models(self):
        """Carga el modelo de TensorFlow y los índices de clases"""
        try:
            if not all(os.path.exists(p) for p in [MODEL_PATH, CLASS_INDICES_PATH]):
                raise FileNotFoundError("No se encontraron los archivos del modelo")
            
            self.model = load_model(MODEL_PATH)
            with open(CLASS_INDICES_PATH, 'r') as f:
                self.class_map = {int(k): v for k, v in json.load(f).items()}
            
            # Configurar MediaPipe
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=0  # Modelo ligero para CPU
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
        except Exception as e:
            print(f"Error al inicializar modelos: {str(e)}")
            self.prediction_text = f"Error: {str(e)}"

    def init_camera(self):
        """Configura la cámara según el sistema operativo"""
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(0, backend)
        
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            self.prediction_text = "Error: Cámara no disponible"
        else:
            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Cámara configurada ({self.cam_width}x{self.cam_height})")

    def init_ui(self):
        """Configura la interfaz gráfica"""
        layout = QVBoxLayout(self)
        
        # Etiqueta para mostrar el video
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label)
        
        # Etiqueta para predicciones
        self.prediction_label = QLabel("Inicializando...", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        font = self.prediction_label.font()
        font.setPointSize(14)
        self.prediction_label.setFont(font)
        layout.addWidget(self.prediction_label)
        
        # Área de texto
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Texto reconocido aparecerá aquí")
        layout.addWidget(self.text_edit)
        
        self.setLayout(layout)

    def update_frame(self):
        """Procesa cada frame de la cámara"""
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Procesamiento con MediaPipe (necesita RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Predicción de gestos
        current_prediction = "Acerca tu mano a la cámara"
        if results.multi_hand_landmarks:
            landmarks = preprocess_landmarks(results.multi_hand_landmarks[0])
            if np.any(landmarks) and self.model:
                prediction = self.model.predict(landmarks.reshape(1, -1), verbose=0)
                confidence = np.max(prediction)
                
                if confidence > 0.8:
                    class_name = self.class_map.get(np.argmax(prediction), "Desconocido")
                    current_prediction = f"{class_name} ({confidence*100:.1f}%)"
                    
                    # Lógica para agregar texto después de 1.5 segundos
                    if current_prediction == self.last_predicted_class:
                        if time.time() - self.start_time >= 1.5:
                            self.text_edit.insertPlainText(class_name[0].upper())
                            self.start_time = time.time()
                    else:
                        self.start_time = time.time()
                        self.last_predicted_class = current_prediction
        
        # Mostrar texto de predicción
        self.prediction_label.setText(current_prediction)
        
        display_frame = frame_rgb.copy()
        #if results.multi_hand_landmarks:
        #    # Convertir temporalmente a BGR para dibujar (MediaPipe usa BGR para dibujar)
         #   frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
         #   for hand_landmarks in results.multi_hand_landmarks:
         #       self.mp_drawing.draw_landmarks(
          #          frame_bgr,
           #         hand_landmarks,
            #        self.mp_hands.HAND_CONNECTIONS
             #   )
            # Volver a RGB para mostrar
            #display_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Mostrar el frame en la interfaz (en RGB)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def closeEvent(self, event):
        """Liberar recursos al cerrar"""
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        event.accept()


if __name__ == "__main__":
    configure_qt()
    
    #QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    #QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    if not os.path.isdir(BASE_DIR):
        print(f"Error: Directorio no encontrado: {BASE_DIR}")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    try:
        window = CameraApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error crítico: {str(e)}")
        sys.exit(1)