import cv2
import mediapipe as mp
import os
import numpy as np
import time


nombre = 'Y_Gesto'
direccion_entrenamiento = 'proyecto/DATA/pro_1/Entrenamiento'
direccion_validacion = 'proyecto/DATA/pro_1/Validacion'
#deteccion de dedos ocultos puede ocurrir variaciones en la deteccion en la letras m y n 
carpeta_entrenamiento = os.path.join(direccion_entrenamiento, nombre)
carpeta_validacion = os.path.join(direccion_validacion, nombre)
os.makedirs(carpeta_entrenamiento, exist_ok=True)
os.makedirs(carpeta_validacion, exist_ok=True)
#j-k-enie-Q-X-Z
#CORREGIR LA N, AGREGAR MÁS DATOS CENTRADO CON LA LETRA U, IMPLEMNTAR LA LETRA Y
cont_entrenamiento = 0
cont_validacion = 0
total_datos = 300 
porcentaje_validacion = 0.1  #para validación cambiar a 0.2 cuando se agrega por primera vez y si no es la primera 0.1
WRIST = 0 # punto cero que seria muneca 
MIDDLE_FINGER_MCP = 9 # Metacarpophalangeal dedo del medio 



cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
last_capture_time = time.time()

def normalizar_landmarks(landmarks, image_shape):
    """
    Normaliza landmarks para que la distancia y lugar de la mano no importe cuando se normalize quede casi los mismos numero auqnue la diferencia de
    para que despues de la resta de la muneca como pi′​=pi​−p p0 coordenadas de la muenca
    pi' landmark trasladado 
    la normalizacion por escala
    para esti se ocupara la base del dedo del medio el punto 9,la d'9 = raiz((x,2)9,(y,2)9,(z,2)9)  que seria raziz de la suma de x,y,z elevados al cuadrado 
    
    todos los landmarks se dividiran por d que seria el resultado de la ecuacion 
    
    """
    if not landmarks or not landmarks.landmark:
        return np.zeros(21 * 3) 

    landmarks_list = landmarks.landmark 

    # Convertir a array NumPy (21 filas, 3 columnas)
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list]) 
    
    # normalizacion por traslacion
    # Usar la muñeca (índice WRIST) como punto de referencia restar a todos los demas el punto de referencia
    reference_point = landmarks_array[WRIST]
    landmarks_relative = landmarks_array - reference_point 
    # Ahora el landmark WRIST tiene coordenadas [0, 0, 0] en 'landmarks_relative'
    
    # normalizacion por excala 
    # Calcular la distancia entre la muñeca (ahora en 0,0,0) y la base del dedo medio que seria obtener las coordenadas del landmark 9
    middle_finger_mcp_coords = landmarks_relative[MIDDLE_FINGER_MCP]
    
    # Distancia Euclidiana desde el origen (0,0,0) la funcion euclidiana es la raiz de la suma de x,y,z elevado al cuadrado
    scale_factor = np.linalg.norm(middle_finger_mcp_coords) 
    
    # Evitar división por cero si la mano está en una pose extraña o muy cerca/lejos
    if scale_factor < 1e-6: 

         return np.zeros(21 * 3)

    # Dividir todas las coordenadas relativas por el factor de escala que seria el resultado obtenido anterior
    landmarks_normalized = landmarks_relative / scale_factor

    return landmarks_normalized.flatten()

try:
    while (cont_entrenamiento + cont_validacion) < total_datos:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks en el frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
                # Captura automática cada 0.5 segundos
                if time.time() - last_capture_time > 0.5:
                    landmarks_vector = normalizar_landmarks(hand_landmarks, frame.shape)
                    
                    if landmarks_vector is not None and np.any(landmarks_vector): # Chequea si no son todos ceros
                        if np.random.rand() < porcentaje_validacion:
                            # Comprobar si el directorio existe antes de guardar
                            if not os.path.exists(carpeta_validacion):
                                os.makedirs(carpeta_validacion)
                            np.save(os.path.join(carpeta_validacion, f"Mano_{cont_validacion}.npy"), landmarks_vector)
                            cont_validacion += 1
                        else:
                            if not os.path.exists(carpeta_entrenamiento):
                                os.makedirs(carpeta_entrenamiento)
                            np.save(os.path.join(carpeta_entrenamiento, f"Mano_{cont_entrenamiento}.npy"), landmarks_vector)
                            cont_entrenamiento += 1
                        
                        last_capture_time = time.time() 
        
        info_text = f"Entrenamiento: {cont_entrenamiento}, Validacion: {cont_validacion}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Captura de Landmarks", frame)

        
        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura completada: {cont_entrenamiento} (entrenamiento), {cont_validacion} (validación)")