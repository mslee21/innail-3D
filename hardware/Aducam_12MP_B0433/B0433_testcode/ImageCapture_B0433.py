import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(0)  # 0번 장치 (필요시 1, 2로 변경)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)

os.makedirs("saved_frames", exist_ok=True)
print("[▶] 실시간 영상 스트리밍 시작 (ESC로 종료, S 키로 저장)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라로부터 영상을 읽을 수 없습니다.")
        break

    cv2.imshow("Live Preview", cv2.resize(frame, (800, 600)))

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        filename = datetime.now().strftime("saved_frames/frame_%Y%m%d_%H%M%S.png")
        cv2.imwrite(filename, frame)
        print(f"[💾] 저장됨: {filename}")

cap.release()
cv2.destroyAllWindows()
print("[🛑] 스트리밍 종료")