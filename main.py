import cv2
import numpy as np
import time
import math
import pyautogui

import os
# TensorFlow / TFLite loglarını azalt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
# Google logging/absl uyarılarını kapat
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_logtostderr"] = "1"

# MediaPipe 0.10.x API
import mediapipe as mp
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles



# ----------------- KONFİG -----------------
CAM_INDEX = 0              # Birden fazla kameranız varsa 1,2 deneyebilirsiniz
FRAME_WIDTH = 1280         # Yakınlaştırma/performans için düşürebilirsiniz
FRAME_HEIGHT = 720

#SMOOTHING_ALPHA = 0.25     # İmleç yumuşatma (0 < a ≤ 1). Daha küçük -> daha pürüzsüz
#PINCH_THRESHOLD = 0.035     # Başparmak–parmak ucu arasındaki normalize mesafe eşiği
#PINCH_HYSTERESIS = 0.005    # Eşik histerezis (stabilite)
#DRAG_HOLD_DELAY = 0.20      # Sol-drag başlamadan önce pinch süresi (sn)
#SCROLL_SENSITIVITY = 1500   # Scroll hızı (dy ile çarpılır)
SHOW_OVERLAY = True         # Landmarks ve FPS gösterimi

SMOOTHING_ALPHA = 0.12
PINCH_THRESHOLD = 0.042
PINCH_HYSTERESIS = 0.007
DRAG_HOLD_DELAY = 0.35
SCROLL_SENSITIVITY = 900


# PyAutoGUI güvenlik
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# ----------------- YARDIMCI FONKSİYONLAR -----------------
def norm_dist(p1, p2):
    """ İki nokta arası öklid mesafesi (normalize koordinatlar: 0..1). """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def ema(prev, new, alpha=SMOOTHING_ALPHA):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------- DURUM DEĞİŞKENLERİ -----------------
prev_mouse_x = None
prev_mouse_y = None

left_pinch_active = False
left_pinch_start_time = 0.0
dragging = False

right_pinch_active = False
scroll_pinch_active = False
prev_scroll_ref_y = None

# Histerezisli eşik hesapları
def below_thresh(d, t=PINCH_THRESHOLD):
    return d < t

def above_release(d, t=PINCH_THRESHOLD + PINCH_HYSTERESIS):
    return d > t

# ----------------- ANA UYGULAMA -----------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    global prev_mouse_x, prev_mouse_y
    global left_pinch_active, left_pinch_start_time, dragging
    global right_pinch_active, scroll_pinch_active, prev_scroll_ref_y

    # MediaPipe Hands
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,           # Tek el ile daha stabil
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    fps_t0 = time.time()
    fps_cnt = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kameradan görüntü alınamadı.")
                break

            # Aynalı görünüm (kullanıcı için doğal)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe tahmini
            res = hands.process(rgb)

            h, w, _ = frame.shape
            now = time.time()

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]  # tek el

                # Landmark'ları liste olarak al (normalize koordinatlar 0..1)
                pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Önemli noktalar (normalize)
                thumb_tip = (pts[4][0], pts[4][1])
                index_tip = (pts[8][0], pts[8][1])
                middle_tip = (pts[12][0], pts[12][1])
                ring_tip = (pts[16][0], pts[16][1])

                # --- İMLEÇ KONTROLÜ (işaret parmağı ucu) ---
                # Normalized (0..1) -> ekran pikseli
                raw_x = index_tip[0] * SCREEN_W
                raw_y = index_tip[1] * SCREEN_H
                # EMA yumuşatma
                prev_mouse_x = ema(prev_mouse_x, raw_x)
                prev_mouse_y = ema(prev_mouse_y, raw_y)
                # Ekran sınırları
                mx = clamp(prev_mouse_x, 0, SCREEN_W - 1)
                my = clamp(prev_mouse_y, 0, SCREEN_H - 1)
                pyautogui.moveTo(mx, my, duration=0)  # anlık, yumuşatmayı biz yaptık

                # --- GESTURE MESAFELERİ ---
                d_thumb_index = norm_dist(thumb_tip, index_tip)
                d_thumb_middle = norm_dist(thumb_tip, middle_tip)
                d_thumb_ring   = norm_dist(thumb_tip, ring_tip)

                # --- SOL TIK / SÜRÜKLE-BIRAK (Başparmak-İşaret pinch) ---
                if not left_pinch_active and below_thresh(d_thumb_index):
                    left_pinch_active = True
                    left_pinch_start_time = now
                elif left_pinch_active and above_release(d_thumb_index):
                    # Bırakıldı
                    left_pinch_active = False
                    # Drag aktifse bırak
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    else:
                        # Kısa pinch -> tek tıklama
                        # (Drag gecikmesinden kısa sürdüyse)
                        if (now - left_pinch_start_time) < DRAG_HOLD_DELAY:
                            pyautogui.click()

                # Uzun pinch -> mouseDown (sürükle)
                if left_pinch_active and not dragging and (now - left_pinch_start_time) >= DRAG_HOLD_DELAY:
                    pyautogui.mouseDown()
                    dragging = True

                # --- SAĞ TIK (Başparmak-Orta pinch) ---
                if not right_pinch_active and below_thresh(d_thumb_middle):
                    right_pinch_active = True
                elif right_pinch_active and above_release(d_thumb_middle):
                    right_pinch_active = False
                    pyautogui.click(button='right')

                # --- SCROLL (Başparmak-Yüzük pinch + dikey hareket) ---
                if not scroll_pinch_active and below_thresh(d_thumb_ring):
                    scroll_pinch_active = True
                    prev_scroll_ref_y = index_tip[1]
                elif scroll_pinch_active and above_release(d_thumb_ring):
                    scroll_pinch_active = False
                    prev_scroll_ref_y = None

                if scroll_pinch_active and prev_scroll_ref_y is not None:
                    dy = index_tip[1] - prev_scroll_ref_y
                    # Negatif dy -> yukarı kaydır (tarayıcı mantığıyla)
                    scroll_amount = int(-dy * SCROLL_SENSITIVITY)
                    if scroll_amount != 0:
                        pyautogui.scroll(scroll_amount)
                        prev_scroll_ref_y = index_tip[1]

                # Görsel çizimler
                if SHOW_OVERLAY:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    # Eşik durumlarını metinle göster
                    cv2.putText(frame, f"LeftPinch:{left_pinch_active} Drag:{dragging}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"RightPinch:{right_pinch_active} ScrollPinch:{scroll_pinch_active}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # FPS hesap/overlay
            fps_cnt += 1
            if now - fps_t0 >= 1.0:
                fps_val = fps_cnt / (now - fps_t0)
                fps_cnt = 0
                fps_t0 = now
            if SHOW_OVERLAY:
                cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("El ile Kontrol - q ile cikis", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
