import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) # İşlenmiş görüntüyü GUI'ye atar
    status_signal = pyqtSignal(str)               # Durum metnini GUI'ye atar

    def __init__(self, processor, cam_index=0):
        super().__init__()
        self.processor = processor # main.py'den gelen mantık işlemcisi
        self.cam_index = cam_index
        self._run_flag = True

    def run(self):
        # Kamera ayarları processor içindeki config'den de alınabilir ama basit tutuyoruz
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_skip = 0
        
        while self._run_flag:
            ok, frame = cap.read()
            if not ok:
                break

            # Frame atlama optimizasyonu (Processor içinde de yapılabilir)
            frame_skip += 1
            if frame_skip % 1 != 0: 
                continue

            # --- TÜM MANTIK BURADA ÇAĞRILIR ---
            # frame gönderilir, işlenmiş frame ve durum metni geri alınır
            processed_frame, status_text = self.processor.process_frame(frame)

            self.change_pixmap_signal.emit(processed_frame)
            self.status_signal.emit(status_text)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class HandUI(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor # Logic sınıfı
        self.thread = None

        self.setWindowTitle("HandDetect - Gesture Control")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Ana Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Kamera Alanı
        self.image_label = QLabel(self)
        self.image_label.resize(640, 360)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Durum Metni
        self.status_label = QLabel("Sistem Hazır", self)
        self.status_label.setStyleSheet("font-size: 16px; color: #00ff00; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

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
            # Thread'e processor'ı veriyoruz
            self.thread = CameraThread(self.processor)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_signal.connect(self.update_status)
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

    def update_image(self, cv_img):
        """OpenCV (BGR) görüntüsünü PyQt (RGB) formatına çevirir."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(640, 360, Qt.KeepAspectRatio))

    def update_status(self, text):
        self.status_label.setText(text)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()