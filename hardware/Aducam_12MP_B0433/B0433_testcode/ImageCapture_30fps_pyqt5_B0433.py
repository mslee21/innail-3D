import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class CameraWindow(QWidget):
    def __init__(self, cam_index=0, capture_width=1280, capture_height=720, display_width=640, display_height=360, fps=30):
        super().__init__()
        self.setWindowTitle("📷 PyQt5 실시간 영상 (FPS + 리사이징 최적화)")

        # 설정
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.fps = fps

        # FPS 측정 변수
        self.last_time = time.time()
        self.frame_count = 0

        # 카메라 설정
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # GUI 구성
        self.image_label = QLabel()
        self.fps_label = QLabel("FPS: --")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        self.setLayout(layout)

        # 타이머로 프레임 갱신
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.fps))

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 리사이징 → RGB 변환
            frame = cv2.resize(frame, (self.display_width, self.display_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # QImage 생성 및 GUI 출력
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))

            # FPS 계산
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                fps = self.frame_count / (now - self.last_time)
                self.fps_label.setText(f"FPS: {fps:.2f}")
                self.last_time = now
                self.frame_count = 0

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraWindow(
        cam_index=0,             # /dev/video0 or index 0
        capture_width=1280,      # 카메라 입력 해상도
        capture_height=720,
        display_width=640,       # GUI 표시 해상도
        display_height=360,
        fps=30
    )
    win.show()
    sys.exit(app.exec_())