import sys
import cv2
import numpy as np
import time
import math
import pyautogui
import os
import platform

# PyQt5 Uygulaması için
from PyQt5.QtWidgets import QApplication
# Arayüz dosyamızdan import (Dosya yapısına uygun)
from ui.interface import HandUI

# Mediapipe ve Log Ayarları
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "3"
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----------------- KONFİGÜRASYON -----------------
SMOOTHING_ALPHA = 0.15
PINCH_THRESHOLD = 0.045
PINCH_HYSTERESIS = 0.012
DRAG_HOLD_DELAY = 0.35
SCROLL_SENSITIVITY = 900
SCROLL_COOLDOWN = 0.08
GESTURE_LOCK_TIME = 0.3
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False

# ----------------- YARDIMCI FONKSİYONLAR -----------------
def norm_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def ema(prev, new, alpha=SMOOTHING_ALPHA):
    if prev is None: return new
    return (1 - alpha) * prev + alpha * new

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def volume_system_control(action):
    """İşletim sistemine göre ses kontrolü yapar."""
    sys_plat = platform.system()
    if sys_plat == "Windows":
        if action == "increase": pyautogui.press("volumeup")
        elif action == "decrease": pyautogui.press("volumedown")
    elif sys_plat == "Darwin": # macOS
        cmd = "set volume output volume (output volume of (get volume settings) + 5)" if action == "increase" else "set volume output volume (output volume of (get volume settings) - 5)"
        os.system(f'osascript -e "{cmd}"')
    elif sys_plat == "Linux":
        sign = "+" if action == "increase" else "-"
        os.system(f"amixer -D pulse sset Master 5%{sign}")

# ----------------- MANTIK İŞLEMCİSİ (PROCESSOR) -----------------
class GestureProcessor:
    """
    Tüm görüntü işleme ve gesture mantığı burada döner.
    UI bu sınıfı sadece 'process_frame' fonksiyonu ile kullanır.
    """
    def __init__(self):
        # Mediapipe Başlat
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Durum Değişkenleri
        self.prev_mouse_x = None
        self.prev_mouse_y = None
        
        self.left_pinch_active = False
        self.left_pinch_start_time = 0.0
        self.dragging = False

        self.right_pinch_active = False
        self.right_pinch_last_click = 0.0

        self.scroll_pinch_active = False
        self.prev_scroll_ref_y = None
        self.last_scroll_time = 0.0

        self.volume_active = False
        self.prev_vol_ref_x = None
        self.last_vol_time = 0.0
        
        self.active_gesture = None
        self.gesture_lock_start = 0.0

    def can_start_gesture(self, gesture_type, now):
        if self.active_gesture is None: return True
        if self.active_gesture == gesture_type: return True
        if now - self.gesture_lock_start > GESTURE_LOCK_TIME: return True
        return False

    def set_active_gesture(self, gesture_type, now):
        if self.active_gesture != gesture_type:
            self.active_gesture = gesture_type
            self.gesture_lock_start = now

    def clear_active_gesture(self):
        self.active_gesture = None

    def process_frame(self, frame):
        """
        Gelen ham frame'i işler, gestureları algılar, mouse/ses kontrolü yapar
        ve üzerine çizim yapılmış frame ile durum metnini döndürür.
        """
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        now = time.time()
        
        status_text = "El Algılanamadı"

        if res.multi_hand_landmarks:
            status_text = "El Takip Ediliyor"
            hand_landmarks = res.multi_hand_landmarks[0]
            pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            h, w, c = frame.shape

            # Parmak Uçları
            thumb_tip = (pts[4][0], pts[4][1])
            index_tip = (pts[8][0], pts[8][1])
            middle_tip = (pts[12][0], pts[12][1])
            ring_tip = (pts[16][0], pts[16][1])
            pinky_tip = (pts[20][0], pts[20][1])

            # --- MOUSE HAREKETİ (Başparmak Ucu - İstenildiği gibi) ---
            raw_x = thumb_tip[0] * SCREEN_W
            raw_y = thumb_tip[1] * SCREEN_H
            self.prev_mouse_x = ema(self.prev_mouse_x, raw_x)
            self.prev_mouse_y = ema(self.prev_mouse_y, raw_y)
            
            # Gesture kilitliyse (ses/scroll) mouse hareket etmesin
            if not self.volume_active and not self.scroll_pinch_active:
                mx = clamp(self.prev_mouse_x, 0, SCREEN_W - 1)
                my = clamp(self.prev_mouse_y, 0, SCREEN_H - 1)
                pyautogui.moveTo(mx, my, duration=0)

            # Mesafeler
            d_thumb_index = norm_dist(thumb_tip, index_tip)
            d_thumb_middle = norm_dist(thumb_tip, middle_tip)
            d_thumb_ring = norm_dist(thumb_tip, ring_tip)
            d_thumb_pinky = norm_dist(thumb_tip, pinky_tip)

            # --- 1. SES KONTROL (Başparmak + Serçe) ---
            if d_thumb_pinky < PINCH_THRESHOLD and self.can_start_gesture("volume", now):
                if not self.volume_active:
                    self.volume_active = True
                    self.prev_vol_ref_x = index_tip[0]
                    self.set_active_gesture("volume", now)
            elif self.volume_active and d_thumb_pinky > (PINCH_THRESHOLD + PINCH_HYSTERESIS):
                self.volume_active = False
                self.prev_vol_ref_x = None
                self.clear_active_gesture()

            if self.volume_active and self.prev_vol_ref_x is not None:
                status_text = "MOD: SES KONTROL"
                if (now - self.last_vol_time) >= 0.15: # Cooldown
                    dx = index_tip[0] - self.prev_vol_ref_x
                    if dx > 0.015: 
                        volume_system_control("increase")
                        self.prev_vol_ref_x = index_tip[0]
                        self.last_vol_time = now
                        status_text = "SES: ARTTIRILIYOR"
                    elif dx < -0.015: 
                        volume_system_control("decrease")
                        self.prev_vol_ref_x = index_tip[0]
                        self.last_vol_time = now
                        status_text = "SES: AZALTILIYOR"

            # --- 2. SCROLL (Başparmak + Yüzük) ---
            elif d_thumb_ring < PINCH_THRESHOLD and self.can_start_gesture("scroll", now):
                if not self.scroll_pinch_active:
                    self.scroll_pinch_active = True
                    self.prev_scroll_ref_y = index_tip[1]
                    self.set_active_gesture("scroll", now)
            elif self.scroll_pinch_active and d_thumb_ring > (PINCH_THRESHOLD + PINCH_HYSTERESIS):
                self.scroll_pinch_active = False
                self.prev_scroll_ref_y = None
                self.clear_active_gesture()

            if self.scroll_pinch_active:
                status_text = "MOD: SCROLL"
                if self.prev_scroll_ref_y is not None and (now - self.last_scroll_time) >= SCROLL_COOLDOWN:
                    dy = index_tip[1] - self.prev_scroll_ref_y
                    scroll_amount = int(-dy * SCROLL_SENSITIVITY)
                    if abs(scroll_amount) > 5:
                        pyautogui.scroll(scroll_amount)
                        self.prev_scroll_ref_y = index_tip[1]
                        self.last_scroll_time = now

            # --- 3. SAĞ TIK (Başparmak + Orta) ---
            elif d_thumb_middle < PINCH_THRESHOLD and self.can_start_gesture("right", now):
                if not self.right_pinch_active:
                    self.right_pinch_active = True
                    self.set_active_gesture("right", now)
            elif self.right_pinch_active and d_thumb_middle > (PINCH_THRESHOLD + PINCH_HYSTERESIS):
                self.right_pinch_active = False
                if (now - self.right_pinch_last_click) > 0.3:
                    pyautogui.click(button='right')
                    self.right_pinch_last_click = now
                    status_text = "ISLEM: SAG TIK"
                self.clear_active_gesture()

            # --- 4. SOL TIK / SÜRÜKLE (Başparmak + İşaret) ---
            elif d_thumb_index < PINCH_THRESHOLD and self.can_start_gesture("left", now):
                if not self.left_pinch_active:
                    self.left_pinch_active = True
                    self.left_pinch_start_time = now
                    self.set_active_gesture("left", now)
            elif self.left_pinch_active and d_thumb_index > (PINCH_THRESHOLD + PINCH_HYSTERESIS):
                self.left_pinch_active = False
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                    status_text = "ISLEM: SURUKLEME BITTI"
                else:
                    if (now - self.left_pinch_start_time) < DRAG_HOLD_DELAY:
                        pyautogui.click()
                        status_text = "ISLEM: SOL TIK"
                self.clear_active_gesture()

            if self.left_pinch_active and not self.dragging and (now - self.left_pinch_start_time) >= DRAG_HOLD_DELAY:
                pyautogui.mouseDown()
                self.dragging = True
                status_text = "MOD: SURUKLEME"

            # --- ÇİZİM ---
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (0, 255, 0)
                if idx == 20 and self.volume_active: color = (255, 0, 0) # Serçe Kırmızı
                if idx == 4: color = (0, 255, 255) # Başparmak (Mouse) Sarı/Turkuaz
                cv2.circle(frame, (cx, cy), 4, color, -1)

        return frame, status_text

# ----------------- UYGULAMA BAŞLANGICI -----------------
def main():
    app = QApplication(sys.argv)
    
    # Logic nesnesini yarat
    processor = GestureProcessor()
    
    # UI'ı yarat ve logic'i enjekte et
    window = HandUI(processor)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()