
import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime

class CameraViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Camera Preview")
        self.resize(900, 700)

        # UI 구성
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.save_btn = QtWidgets.QPushButton("📷 캡처/저장")
        self.save_btn.clicked.connect(self.save_frame)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

        # 저장 폴더 준비
        os.makedirs("saved_frames", exist_ok=True)

        # 카메라 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)

        # 타이머로 프레임 업데이트
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        print("[▶] 스트리밍 시작")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame

        # OpenCV → Qt 이미지로 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(800, 600, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def save_frame(self):
        if self.current_frame is not None:
            filename = datetime.now().strftime("saved_frames/frame_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, self.current_frame)
            print(f"[💾] 저장됨: {filename}")

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        print("[🛑] 스트리밍 종료")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = CameraViewer()
    viewer.show()
    sys.exit(app.exec_())
