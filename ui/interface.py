import platform
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Kamera ayarları
CAM_W, CAM_H, CAM_FPS = 640, 360, 30
PROCESS_EVERY_N_FRAMES = 1


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    gesture_signal = pyqtSignal(str)

    def __init__(self, processor, cam_index=0):
        super().__init__()
        self.processor = processor
        self.cam_index = cam_index
        self._run_flag = True

    def run(self):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cam_index)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

        if not cap.isOpened():
            self.status_signal.emit("Kamera açılamadı")
            return

        frame_skip = 0

        while self._run_flag:
            ok, frame = cap.read()
            if not ok:
                self.status_signal.emit("Kamera okunamıyor")
                break

            frame_skip += 1
            if (frame_skip % PROCESS_EVERY_N_FRAMES) != 0:
                continue

            # >>> 3 değer bekliyoruz
            processed_frame, status_text, gesture_text = self.processor.process_frame(frame)

            self.change_pixmap_signal.emit(processed_frame)
            self.status_signal.emit(status_text)
            self.gesture_signal.emit(gesture_text)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class HandUI(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.thread = None

        self.setWindowTitle("HandDetect - Gesture Control")
        self.setGeometry(100, 100, 860, 700)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Kamera alanı
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(CAM_W, CAM_H)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Durum
        self.status_label = QLabel("Sistem Hazır", self)
        self.status_label.setStyleSheet("font-size: 16px; color: #00ff00; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Son hareket
        self.gesture_label = QLabel("Son Hareket: -", self)
        self.gesture_label.setStyleSheet("font-size: 20px; color: #ffd700; font-weight: bold;")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.gesture_label)

        # Butonlar
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setStyleSheet("background-color: #28a745; padding: 10px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_camera)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setStyleSheet("background-color: #dc3545; padding: 10px; border-radius: 5px;")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        self.layout.addLayout(btn_layout)

    def start_camera(self):
        if self.thread is None:
            self.thread = CameraThread(self.processor, cam_index=0)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_signal.connect(self.update_status)
            self.thread.gesture_signal.connect(self.update_gesture)
            self.thread.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

    def stop_camera(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear()
        self.status_label.setText("Kamera Durduruldu")
        self.gesture_label.setText("Son Hareket: -")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                CAM_W, CAM_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def update_status(self, text):
        self.status_label.setText(text)

    def update_gesture(self, text):
        self.gesture_label.setText(f"Son Hareket: {text}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
