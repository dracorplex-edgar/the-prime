import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import sys
import os 


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


def main():
    try:
        # Cargar modelo y clases
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")
        if not os.path.exists(CLASS_INDICES_PATH):
             raise FileNotFoundError(f"No se encontró el archivo de índices de clase en: {CLASS_INDICES_PATH}")

        model = load_model(MODEL_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_map = {int(k): v for k, v in class_indices.items()} # Mapeo para predicción

        # Configurar MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Verifica si está siendo usada por otra aplicación.")

        # Configurar ventana
        window_name = 'Detección de Gestos'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False # Optimización: Marcar como no escribible
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True # Marcar como escribible de nuevo

            prediction_text = "No se detecta mano" # Texto por defecto

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    #funcion para normalizacion de landmarks
                    landmarks_vector = preprocess_landmarks(hand_landmarks, frame.shape)

                    # Solo predecir si la normalización funciona
                    if np.any(landmarks_vector):
                        landmarks_vector = landmarks_vector.reshape(1, -1) # Asegurar forma (1, 63) para el modelo

                        prediction = model.predict(landmarks_vector, verbose=0)
                        predicted_class_index = np.argmax(prediction[0]) # Obtener índice
                        confidence = np.max(prediction[0]) # Obtener confianza

                        if confidence > 0.8: # Umbral de confianza maniobrar si quiere mejor exactitud
                            predicted_class_name = class_map.get(predicted_class_index, "no reconocible") # Obtener nombre
                            prediction_text = f"{predicted_class_name} ({confidence*100:.1f}%)"
                        else:
                            prediction_text = "Gesto no reconocido"
                    else:
                        prediction_text = "Error en normalización" 

            # Mostrar resultado (fuera del bucle for de manos, ya que solo hay una)
            cv2.putText(
                frame,
                prediction_text,
                (10, 50), # Posición del texto
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA 
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF # Usar máscara para compatibilidad
            if (key == 27 or
                key == ord('q') or
                cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
                break

    except FileNotFoundError as e:
        print(f"Error de archivo: {e}")
    except RuntimeError as e:
        print(f"Error de ejecución: {e}")
    except Exception as e:
        print(f"Error inesperado: {type(e).__name__} - {str(e)}")
    finally:
        print("Cerrando aplicación y liberando recursos...")
        if 'hands' in locals() and hasattr(hands, 'close'):
            hands.close()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()