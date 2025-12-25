import sys
import os
import time
import math
import platform

import cv2
import pyautogui
import mediapipe as mp

from PyQt5.QtWidgets import QApplication
from ui.interface import HandUI

# Mediapipe log azalt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "3"

mp_hands = mp.solutions.hands

# ----------------- KONFİG -----------------
CURSOR_DEADZONE_PX = 2
CURSOR_MAX_STEP_PX = 90
USE_BLEND_POINT = True

ONEEURO_MIN_CUTOFF = 1.2
ONEEURO_BETA = 0.015
ONEEURO_D_CUTOFF = 1.0

PINCH_HYSTERESIS_RATIO = 0.08
DRAG_HOLD_DELAY = 0.35
SCROLL_SENSITIVITY = 900
SCROLL_COOLDOWN = 0.08
GESTURE_LOCK_TIME = 0.30

VOLUME_COOLDOWN = 0.15
VOLUME_STEP_DX = 0.015

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


# ----------------- YARDIMCI -----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def volume_system_control(action):
    sys_plat = platform.system()
    if sys_plat == "Windows":
        pyautogui.press("volumeup" if action == "increase" else "volumedown")
    elif sys_plat == "Darwin":
        cmd = (
            "set volume output volume (output volume of (get volume settings) + 5)"
            if action == "increase"
            else "set volume output volume (output volume of (get volume settings) - 5)"
        )
        os.system(f'osascript -e "{cmd}"')
    elif sys_plat == "Linux":
        sign = "+" if action == "increase" else "-"
        os.system(f"amixer -D pulse sset Master 5%{sign}")


# ----------------- One Euro Filter -----------------
class LowPass:
    def __init__(self):
        self.y = None

    def filt(self, x, a):
        if self.y is None:
            self.y = x
            return x
        self.y = a * x + (1 - a) * self.y
        return self.y

def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(1e-6, dt))

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPass()
        self.dx_filter = LowPass()
        self.last_t = None
        self.last_x = None

    def __call__(self, x, t):
        if self.last_t is None:
            self.last_t = t
            self.last_x = x
            self.x_filter.y = x
            self.dx_filter.y = 0.0
            return x

        dt = max(1e-6, t - self.last_t)
        self.last_t = t

        dx = (x - self.last_x) / dt
        self.last_x = x

        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_filter.filt(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        return self.x_filter.filt(x, a)


# ----------------- PROCESSOR -----------------
class GestureProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.60
        )

        self.fx = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)
        self.fy = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)

        self.prev_mx = None
        self.prev_my = None

        # Gesture state
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

        # UI için "en son hareket"
        self.last_gesture_text = "Bekleniyor"

        # Metrikler
        self.metrics = {
            "fps": 0.0,
            "latency": 0.0,
            "detection_confidence": 0.0,
            "tracking_confidence": 0.0,
            "hand_visible": False,
            "velocity": 0.0,
            "distance_z": 0.0,
            "stability": 100.0,
            "angle": 0.0,
            "stats": {
                "clicks": 0,
                "right_clicks": 0,
                "drags": 0,
                "scroll_amount": 0,
                "vol_steps": 0
            }
        }
        self.frame_times = []
        self._last_raw_x = None
        self._last_raw_y = None

    def _update_fps(self, now):
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            dt = self.frame_times[-1] - self.frame_times[0]
            if dt > 0:
                self.metrics["fps"] = (len(self.frame_times) - 1) / dt

    def can_start_gesture(self, gesture_type, now):
        if self.active_gesture is None:
            return True
        if self.active_gesture == gesture_type:
            return True
        return (now - self.gesture_lock_start) > GESTURE_LOCK_TIME

    def set_active_gesture(self, gesture_type, now):
        if self.active_gesture != gesture_type:
            self.active_gesture = gesture_type
            self.gesture_lock_start = now

    def clear_active_gesture(self):
        self.active_gesture = None

    def _move_cursor_stable(self, x, y, now):
        raw_x, raw_y = x, y
        
        # Velocity calculation
        if self._last_raw_x is not None and self.frame_times:
            dt = max(1e-6, now - self.frame_times[-1]) if len(self.frame_times) > 1 else 0.033
            dist = math.hypot(raw_x - self._last_raw_x, raw_y - self._last_raw_y)
            self.metrics["velocity"] = dist / dt
        
        self._last_raw_x, self._last_raw_y = raw_x, raw_y

        # Filtering
        ux = self.fx(raw_x, now)
        uy = self.fy(raw_y, now)

        # Stability Metric: Comparison of raw movement vs filtered movement
        # Small jitter in raw should be absorbed by filter.
        raw_move = math.hypot(raw_x - (self.prev_mx or raw_x), raw_y - (self.prev_my or raw_y))
        filt_move = math.hypot(ux - (self.prev_mx or ux), uy - (self.prev_my or uy))
        if raw_move > 0.1:
            # Stability is high if filt_move is smooth compared to raw_move
            ratio = clamp(filt_move / raw_move, 0, 1)
            self.metrics["stability"] = 100 * (1.0 - abs(raw_move - filt_move) / (raw_move + 1e-6))
        
        ux = clamp(ux, 0, SCREEN_W - 1)
        uy = clamp(uy, 0, SCREEN_H - 1)

        if self.prev_mx is None:
            self.prev_mx, self.prev_my = ux, uy
            pyautogui.moveTo(ux, uy, duration=0)
            return

        dx = ux - self.prev_mx
        dy = uy - self.prev_my

        if abs(dx) < CURSOR_DEADZONE_PX:
            ux = self.prev_mx
        if abs(dy) < CURSOR_DEADZONE_PX:
            uy = self.prev_my

        step = math.hypot(ux - self.prev_mx, uy - self.prev_my)
        if step > CURSOR_MAX_STEP_PX:
            r = CURSOR_MAX_STEP_PX / max(1e-6, step)
            ux = self.prev_mx + (ux - self.prev_mx) * r
            uy = self.prev_my + (uy - self.prev_my) * r

        self.prev_mx, self.prev_my = ux, uy
        pyautogui.moveTo(ux, uy, duration=0)

    def _reset_if_hand_lost(self):
        if self.dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        self.dragging = False
        self.left_pinch_active = False
        self.right_pinch_active = False
        self.scroll_pinch_active = False
        self.volume_active = False
        self.prev_scroll_ref_y = None
        self.prev_vol_ref_x = None
        self.clear_active_gesture()
        self.last_gesture_text = "Bekleniyor"
        self.metrics["hand_visible"] = False
        self.metrics["velocity"] = 0.0
        self._last_raw_x = None
        self._last_raw_y = None

    def process_frame(self, frame):
        t_start = time.time()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        now = time.time()

        self._update_fps(now)

        status_text = "El Algılanamadı"
        gesture_text = self.last_gesture_text

        if not res.multi_hand_landmarks:
            self._reset_if_hand_lost()
            self.metrics["latency"] = (time.time() - t_start) * 1000
            return frame, status_text, gesture_text, self.metrics

        self.metrics["hand_visible"] = True
        if res.multi_handedness:
            self.metrics["detection_confidence"] = res.multi_handedness[0].classification[0].score * 100

        hand_landmarks = res.multi_hand_landmarks[0]
        pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        h, w, _ = frame.shape

        wrist = (pts[0][0], pts[0][1])
        thumb_tip = (pts[4][0], pts[4][1])
        index_mcp = (pts[5][0], pts[5][1])
        index_tip = (pts[8][0], pts[8][1])
        middle_mcp = (pts[9][0], pts[9][1])
        middle_tip = (pts[12][0], pts[12][1])
        ring_tip = (pts[16][0], pts[16][1])
        pinky_tip = (pts[20][0], pts[20][1])

        # Distance calculation (Z proxy)
        dist_px = norm_dist(pts[0], pts[9]) # Wrist to Middle MCP distance in norm coords
        self.metrics["distance_z"] = clamp(dist_px * 500, 0, 100) # Arbitrary scaling for 0-100 range

        # Angle Calculation
        dy_angle = pts[12][1] - pts[0][1]
        dx_angle = pts[12][0] - pts[0][0]
        self.metrics["angle"] = math.degrees(math.atan2(-dy_angle, dx_angle))

        if USE_BLEND_POINT:
            nx = 0.65 * thumb_tip[0] + 0.35 * index_mcp[0]
            ny = 0.65 * thumb_tip[1] + 0.35 * index_mcp[1]
        else:
            nx, ny = thumb_tip[0], thumb_tip[1]

        raw_x = nx * SCREEN_W
        raw_y = ny * SCREEN_H

        hand_scale = max(1e-6, norm_dist(wrist, middle_mcp))
        pinch_th = clamp(0.42 * hand_scale, 0.030, 0.075)
        pinch_release = pinch_th * (1.0 + PINCH_HYSTERESIS_RATIO)

        d_thumb_index = norm_dist(thumb_tip, index_tip)
        d_thumb_middle = norm_dist(thumb_tip, middle_tip)
        d_thumb_ring = norm_dist(thumb_tip, ring_tip)
        d_thumb_pinky = norm_dist(thumb_tip, pinky_tip)

        if not self.volume_active and not self.scroll_pinch_active:
            self._move_cursor_stable(raw_x, raw_y, now)

        status_text = "El Takip Ediliyor"

        # 1) SES
        if d_thumb_pinky < pinch_th and self.can_start_gesture("volume", now):
            if not self.volume_active:
                self.volume_active = True
                self.prev_vol_ref_x = index_tip[0]
                self.set_active_gesture("volume", now)
                gesture_text = "Ses Modu"
        elif self.volume_active and d_thumb_pinky > pinch_release:
            self.volume_active = False
            self.prev_vol_ref_x = None
            self.clear_active_gesture()

        if self.volume_active and self.prev_vol_ref_x is not None:
            status_text = "MOD: SES KONTROL"
            if (now - self.last_vol_time) >= VOLUME_COOLDOWN:
                dx = index_tip[0] - self.prev_vol_ref_x
                if dx > VOLUME_STEP_DX:
                    volume_system_control("increase")
                    self.metrics["stats"]["vol_steps"] += 1
                    self.prev_vol_ref_x = index_tip[0]
                    self.last_vol_time = now
                    status_text = "SES: ARTTIRILIYOR"
                    gesture_text = "Ses Arttır"
                elif dx < -VOLUME_STEP_DX:
                    volume_system_control("decrease")
                    self.metrics["stats"]["vol_steps"] += 1
                    self.prev_vol_ref_x = index_tip[0]
                    self.last_vol_time = now
                    status_text = "SES: AZALTILIYOR"
                    gesture_text = "Ses Azalt"

        # 2) SCROLL
        elif d_thumb_ring < pinch_th and self.can_start_gesture("scroll", now):
            if not self.scroll_pinch_active:
                self.scroll_pinch_active = True
                self.prev_scroll_ref_y = index_tip[1]
                self.set_active_gesture("scroll", now)
                gesture_text = "Scroll Modu"
        elif self.scroll_pinch_active and d_thumb_ring > pinch_release:
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
                    self.metrics["stats"]["scroll_amount"] += abs(scroll_amount)
                    self.prev_scroll_ref_y = index_tip[1]
                    self.last_scroll_time = now
                    gesture_text = "Scroll"

        # 3) SAĞ TIK
        elif d_thumb_middle < pinch_th and self.can_start_gesture("right", now):
            if not self.right_pinch_active:
                self.right_pinch_active = True
                self.set_active_gesture("right", now)
        elif self.right_pinch_active and d_thumb_middle > pinch_release:
            self.right_pinch_active = False
            if (now - self.right_pinch_last_click) > 0.30:
                pyautogui.click(button="right")
                self.metrics["stats"]["right_clicks"] += 1
                self.right_pinch_last_click = now
                status_text = "ISLEM: SAG TIK"
                gesture_text = "Sağ Tıklandı"
            self.clear_active_gesture()

        # 4) SOL TIK / SÜRÜKLE
        elif d_thumb_index < pinch_th and self.can_start_gesture("left", now):
            if not self.left_pinch_active:
                self.left_pinch_active = True
                self.left_pinch_start_time = now
                self.set_active_gesture("left", now)
        elif self.left_pinch_active and d_thumb_index > pinch_release:
            self.left_pinch_active = False
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
                self.metrics["stats"]["drags"] += 1
                status_text = "ISLEM: SURUKLEME BITTI"
                gesture_text = "Sürükleme Bitti"
            else:
                if (now - self.left_pinch_start_time) < DRAG_HOLD_DELAY:
                    pyautogui.click()
                    self.metrics["stats"]["clicks"] += 1
                    status_text = "ISLEM: SOL TIK"
                    gesture_text = "Sol Tıklandı"
            self.clear_active_gesture()

        if self.left_pinch_active and not self.dragging and (now - self.left_pinch_start_time) >= DRAG_HOLD_DELAY:
            pyautogui.mouseDown()
            self.dragging = True
            status_text = "MOD: SURUKLEME"
            gesture_text = "Sürükleme"

        # UI için en son hareketi sakla
        self.last_gesture_text = gesture_text

        # Hafif çizim
        important = [0, 4, 5, 8, 12, 16, 20, 9]
        for idx in important:
            lm = hand_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 0)
            if idx == 20 and self.volume_active: color = (255, 0, 0)
            if idx == 16 and self.scroll_pinch_active: color = (255, 255, 0)
            if idx == 4: color = (0, 255, 255)
            cv2.circle(frame, (cx, cy), 5, color, -1)

        self.metrics["latency"] = (time.time() - t_start) * 1000
        return frame, status_text, gesture_text, self.metrics



def main():
    app = QApplication(sys.argv)
    processor = GestureProcessor()
    window = HandUI(processor)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
