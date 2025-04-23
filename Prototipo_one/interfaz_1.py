import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

WRIST = 0
MIDDLE_FINGER_MCP = 9 # Metacarpophalangeal joint of the middle finger

BASE_DIR = 'proyecto/src/Prototipo_one/modelo' # directorio base
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_gestos_landmarks.keras')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')

#normalizacion , se cambia cuando el metodo de normarlizacion en la captura lo cambias
def preprocess_landmarks(hand_landmarks, image_shape):
    """
    Preprocesa/Normaliza landmarks para ser invariantes a la traslación
    y a la escala (tamaño/distancia). Usa la muñeca como origen y la
    distancia muñeca-base_dedo_medio como factor de escala.
    """
    if not hand_landmarks or not hand_landmarks.landmark:
        return np.zeros(21 * 3)

    landmarks_list = hand_landmarks.landmark # lista de landmarks

    # Convertir a array NumPy (21 filas, 3 columnas)
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])

    # --- 1. Normalización por Traslación --- investigacion previa prueba
    reference_point = landmarks_array[WRIST].copy() # .copy recomendado por variaciones
    landmarks_relative = landmarks_array - reference_point
    # --- 2. Normalización por Escala ---prueba
    middle_finger_mcp_coords = landmarks_relative[MIDDLE_FINGER_MCP]
    scale_factor = np.linalg.norm(middle_finger_mcp_coords)
    
    if scale_factor < 1e-6:
        return np.zeros(21 * 3)
    landmarks_normalized = landmarks_relative / scale_factor
    return landmarks_normalized.flatten()


# Configurar el entorno para Qt por que si no no agarra el porque segun paquetes "Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/path/to/your/qt/plugins'  # Ajusta esto

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interfaz con Cámara y Texto")
        self.setGeometry(100, 100, 800, 600)
        
        # Configurar la cámara
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print("No se pudo abrir la cámara")
            sys.exit(1)
        
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.init_ui()
        
        # Temporizador para actualizar la camara
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Configurar la etiqueta de la cámara
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label, stretch=4)  
        
        # Configurar el cuadro de texto
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Escribe aquí...")
        self.text_edit.setMaximumHeight(100)  # Altura fija para el cuadro de texto
        layout.addWidget(self.text_edit, stretch=1)  # Menor proporción para el texto
        
        self.setLayout(layout)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir el frame a RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Escalar manteniendo la relación de aspecto
            self.camera_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.width(),
                    self.camera_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
    
    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())