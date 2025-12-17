import cv2
import time

cap = cv2.VideoCapture(0)  # or your cam_index
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

prev = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    cv2.imshow("Test", frame)

    if frame_count >= 30:
        now = time.time()
        fps = frame_count / (now - prev)
        print(f"FPS: {fps:.2f}")
        prev = now
        frame_count = 0

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()