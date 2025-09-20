import cv2
import time
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import csv
import os

# =========================
# Configurações iniciais
# =========================
WINDOW_NAME = "Detecção Facial + Diário Emocional (OpenCV + MediaPipe)"
LOG_PATH = "emotional_log.csv"

# Índices úteis do FaceMesh (MediaPipe)
LIP_LEFT = 61
LIP_RIGHT = 291
LIP_TOP = 13
LIP_BOTTOM = 14
NOSE_TIP = 1
CHIN = 152

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# =========================
# Funções geométricas
# =========================
def euclid_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def mouth_aspect_ratio(landmarks):
    v = euclid_dist(landmarks[LIP_TOP], landmarks[LIP_BOTTOM])
    h = euclid_dist(landmarks[LIP_LEFT], landmarks[LIP_RIGHT])
    if h == 0:
        return 0.0
    return v / h

def smile_score(landmarks):
    y_left = landmarks[LIP_LEFT][1]
    y_right = landmarks[LIP_RIGHT][1]
    y_corners_mean = (y_left + y_right) / 2.0
    y_center = (landmarks[LIP_TOP][1] + landmarks[LIP_BOTTOM][1]) / 2.0
    return y_center - y_corners_mean

def bbox_from_landmarks(landmarks, frame_w, frame_h, pad=0.02):
    xs = [p[0] for p in landmarks.values()]
    ys = [p[1] for p in landmarks.values()]
    xmin = max(0, int((min(xs) - pad) * frame_w))
    xmax = min(frame_w - 1, int((max(xs) + pad) * frame_w))
    ymin = max(0, int((min(ys) - pad) * frame_h))
    ymax = min(frame_h - 1, int((max(ys) + pad) * frame_h))
    return xmin, ymin, xmax, ymax

# =========================
# Classificação de emoção
# =========================
def eyebrow_frown_score(landmarks):
    left = landmarks[70]   # sobrancelha esq
    right = landmarks[300] # sobrancelha dir
    return euclid_dist(left, right)

def classify_emotion(mar, smile, params, landmarks=None):
    if (smile > params['TH_SMILE']) and (mar < params['TH_MOUTH_OPEN_LOW']):
        return "feliz"
    
    if landmarks is not None:
        frown = eyebrow_frown_score(landmarks)
        if (frown < 0.15) and (smile < 0.005) and (mar < params['TH_MOUTH_OPEN_LOW']):
            return "raiva"
    
    if (smile < -params['TH_FROWN']) or (mar > params['TH_MOUTH_OPEN_HIGH']):
        return "negativo"
    
    return "neutro"

# =========================
# Trackbars (parâmetros)
# =========================
def on_trackbar_change(_):
    pass

def create_trackbars(params):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)
    cv2.createTrackbar("TH_SMILE x1000", WINDOW_NAME, int(params['TH_SMILE']*1000), 50, on_trackbar_change)
    cv2.createTrackbar("TH_FROWN x1000", WINDOW_NAME, int(params['TH_FROWN']*1000), 50, on_trackbar_change)
    cv2.createTrackbar("MOUTH_OPEN_LOW x1000", WINDOW_NAME, int(params['TH_MOUTH_OPEN_LOW']*1000), 100, on_trackbar_change)
    cv2.createTrackbar("MOUTH_OPEN_HIGH x1000", WINDOW_NAME, int(params['TH_MOUTH_OPEN_HIGH']*1000), 200, on_trackbar_change)
    cv2.createTrackbar("JANELA_SEG", WINDOW_NAME, params['WINDOW_SEC'], 10, on_trackbar_change)
    cv2.createTrackbar("NEG_RATIO %", WINDOW_NAME, int(params['NEG_RATIO']*100), 100, on_trackbar_change)
    cv2.createTrackbar("ALERTA_MS", WINDOW_NAME, params['ALERT_MIN_MS'], 5000, on_trackbar_change)
    cv2.createTrackbar("LOG_INT seg", WINDOW_NAME, params['LOG_INTERVAL_SEC'], 30, on_trackbar_change)

def read_trackbars():
    th_smile = cv2.getTrackbarPos("TH_SMILE x1000", WINDOW_NAME) / 1000.0
    th_frown = cv2.getTrackbarPos("TH_FROWN x1000", WINDOW_NAME) / 1000.0
    mar_low = cv2.getTrackbarPos("MOUTH_OPEN_LOW x1000", WINDOW_NAME) / 1000.0
    mar_high = cv2.getTrackbarPos("MOUTH_OPEN_HIGH x1000", WINDOW_NAME) / 1000.0
    win_sec = max(1, cv2.getTrackbarPos("JANELA_SEG", WINDOW_NAME))
    neg_ratio = cv2.getTrackbarPos("NEG_RATIO %", WINDOW_NAME) / 100.0
    alert_ms = cv2.getTrackbarPos("ALERTA_MS", WINDOW_NAME)
    log_int = max(1, cv2.getTrackbarPos("LOG_INT seg", WINDOW_NAME))
    return {
        'TH_SMILE': th_smile,
        'TH_FROWN': th_frown,
        'TH_MOUTH_OPEN_LOW': mar_low,
        'TH_MOUTH_OPEN_HIGH': mar_high,
        'WINDOW_SEC': win_sec,
        'NEG_RATIO': neg_ratio,
        'ALERT_MIN_MS': alert_ms,
        'LOG_INTERVAL_SEC': log_int
    }

# =========================
# Main
# =========================
def main():
    params = {
        'TH_SMILE': 0.010,
        'TH_FROWN': 0.010,
        'TH_MOUTH_OPEN_LOW': 0.30,
        'TH_MOUTH_OPEN_HIGH': 0.60,
        'WINDOW_SEC': 5,
        'NEG_RATIO': 0.60,
        'ALERT_MIN_MS': 1500,
        'LOG_INTERVAL_SEC': 5
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    create_trackbars(params)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    last_log_time = time.time()
    emotion_buffer = deque(maxlen=params['WINDOW_SEC'] * 30)  
    emotion_counter = Counter()
    session_start = time.time()
    last_alert_start = None
    alert_active = False

    try:
        while True:
            params = read_trackbars()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = face_mesh.process(frame_rgb)

            emotion_shown = "sem rosto"
            mar, sm = 0.0, 0.0
            bbox = None
            lm = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm = {i: (p.x, p.y) for i, p in enumerate(face_landmarks.landmark)}

                mar = mouth_aspect_ratio(lm)
                sm = smile_score(lm)

                emotion = classify_emotion(mar, sm, params, lm)
                emotion_shown = emotion
                emotion_buffer.append(emotion)

                # Desenhar pontos principais
                key_pts = [LIP_LEFT, LIP_RIGHT, LIP_TOP, LIP_BOTTOM, NOSE_TIP, CHIN]
                for idx in key_pts:
                    cx, cy = int(lm[idx][0]*w), int(lm[idx][1]*h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

                x1, y1, x2, y2 = bbox_from_landmarks(lm, w, h)
                bbox = (x1, y1, x2, y2)

                # Overlay MAR/Smile
                cv2.putText(frame, f"MAR: {mar:.3f} | SMILE: {sm:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

            # Cores por emoção
            colors = {
                "feliz": (0, 255, 0),
                "neutro": (200, 200, 200),
                "negativo": (0, 165, 255),
                "raiva": (0, 0, 255),
                "sem rosto": (100, 100, 100)
            }
            color = colors.get(emotion_shown, (255, 255, 255))

            # Mostrar emoção no canto
            cv2.putText(frame, f"EMOCAO: {emotion_shown.upper()}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Retângulo colorido
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Banner especial para raiva
            if emotion_shown == "raiva":
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), -1)
                cv2.putText(frame, "ALERTA: sinais de RAIVA detectados!",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)

            # Log
            now = time.time()
            if now - last_log_time >= params['LOG_INTERVAL_SEC']:
                if emotion_shown in ["feliz", "neutro", "raiva", "negativo"]:
                    emotion_counter[emotion_shown] += 1
                last_log_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Salvar CSV
        total_entries = sum(emotion_counter.values())
        session_time = time.time() - session_start
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True) if os.path.dirname(LOG_PATH) else None
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["session_seconds", f"{session_time:.1f}"])
            writer.writerow(["samples_logged", total_entries])
            for emo in ["feliz", "neutro", "raiva", "negativo"]:
                writer.writerow([f"count_{emo}", emotion_counter.get(emo, 0)])
        print(f"Diário salvo em: {LOG_PATH}")
        print("Encerrado.")
        
if __name__ == "__main__":
    main()
