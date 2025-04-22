import cv2
import mediapipe as mp
import os
import numpy as np
import time


nombre = 'A_Gesto'
direccion_entrenamiento = 'proyecto/DATA/pro_1/Entrenamiento'
direccion_validacion = 'proyecto/DATA/pro_1/Validacion'

carpeta_entrenamiento = os.path.join(direccion_entrenamiento, nombre)
carpeta_validacion = os.path.join(direccion_validacion, nombre)
os.makedirs(carpeta_entrenamiento, exist_ok=True)
os.makedirs(carpeta_validacion, exist_ok=True)


cont_entrenamiento = 0
cont_validacion = 0
total_datos = 300  
porcentaje_validacion = 0.2  #para validación
padding_bbox = 30


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error :no se puede abrir la camara ")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
#mp_drawing = mp.solutions.drawing_utils # si se quiere dibujar los ladnmarks
last_capture_time = time.time()


try:
    while (cont_entrenamiento + cont_validacion) < total_datos:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la cámara.")
            break

        h,w,_ = frame.shape
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable=False
        results = hands.process(frame_rgb)
        #frame_rgb.flags.writeable = True # Reactivar escritura si es necesario después
        mano_recortada = None
        bbox = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                min_x, min_y = w, h
                max_x, max_y = 0, 0
                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    min_x = min(min_x, px)
                    min_y = min(min_y, py)
                    max_x = max(max_x, px)
                    max_y = max(max_y, py)

                min_x = max(0, min_x - padding_bbox)
                min_y = max(0, min_y - padding_bbox)
                max_x = min(w - 1, max_x + padding_bbox)
                max_y = min(h - 1, max_y + padding_bbox)

                bbox = (min_x, min_y, max_x, max_y)

                # recortar area de la mano
                # Solo si el bounding box tiene un tamaño válido
                if max_x > min_x and max_y > min_y:
                    mano_recortada = frame[min_y:max_y, min_x:max_x].copy()


                current_time = time.time()
                if (current_time - last_capture_time > 0.5) and (mano_recortada is not None) and mano_recortada.size > 0 :

                    if np.random.rand() < porcentaje_validacion:
                        save_folder = carpeta_validacion
                        cont_validacion += 1
                        count = cont_validacion
                        tipo_set = "VAL"
                    else:
                        save_folder = carpeta_entrenamiento
                        cont_entrenamiento += 1
                        count = cont_entrenamiento
                        tipo_set = "ENT"

                    img_name = f"Mano_{nombre}_{tipo_set}_{count}.jpg"
                    img_path = os.path.join(save_folder, img_name)

                    cv2.imwrite(img_path, mano_recortada)

                    last_capture_time = current_time 

        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Mostrar información de conteo en el frame
        info_text = f"ENT: {cont_entrenamiento}, VAL: {cont_validacion} / {total_datos}"
        cv2.putText(frame, f"Capturando Gesto: {nombre}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el frame con las anotaciones
        cv2.imshow("Captura de Imagenes para CNN", frame)

        # --- Salida ---
        # Esperar tecla, salir con ESC (código 27)
        if cv2.waitKey(5) & 0xFF == 27:
            print("(!) Captura interrumpida por el usuario.")
            break
        


finally:
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print(f"Captura completada: {cont_entrenamiento} (entrenamiento), {cont_validacion} (validación)")