import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
from collections import deque

def find_available_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

class CameraApp:
    def __init__(self, window, width=800, height=600, fps=30):
        self.window = window
        self.window.title("📷 실시간 영상 미리보기 (FPS 표시)")
        self.width = width
        self.height = height
        self.fps = fps

        # 카메라 자동 탐색
        self.cam_index = find_available_camera()
        if self.cam_index == -1:
            raise RuntimeError("사용 가능한 카메라가 없습니다.")

        print(f"✅ 사용 중인 카메라 인덱스: {self.cam_index}")

        # FPS 측정용
        self.frame_times = deque(maxlen=10)
        self.last_time = time.time()

        # OpenCV VideoCapture 설정
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # GUI 구성
        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        self.fps_label = tk.Label(self.window, text="FPS: --", font=("Helvetica", 14), fg="blue")
        self.fps_label.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # BGR → RGB 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

            # FPS 측정
            now = time.time()
            self.frame_times.append(now)
            # 1초 내의 프레임 수만 유지
            while len(self.frame_times) > 1 and now - self.frame_times[0] > 1.0:
                self.frame_times.popleft()
            if len(self.frame_times) >= 2:
                duration = self.frame_times[-1] - self.frame_times[0]
                if duration > 0:
                    fps = len(self.frame_times) / duration
                    self.fps_label.configure(text=f"FPS: {fps:.2f}")
                    print(f"[INFO] FPS: {fps:.2f}")  # ← 터미널에 출력!
                else:
                    self.fps_label.configure(text="FPS: ...")
                    print("[INFO] FPS: 계산 불가 (duration == 0)")
            else:
                self.fps_label.configure(text="FPS: ...")
                print("[INFO] FPS: 프레임 누적 중...")

        # 다음 프레임 예약
        self.window.after(int(1000 / self.fps), self.update_frame)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, width=960, height=540, fps=30)
    root.mainloop()
