import cv2

# 카메라 장치 번호 (보통 0 또는 1)
camera_index = 0

# 카메라 캡처 객체 생성
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Windows 환경에서는 CAP_DSHOW 권장

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 최대 해상도 지원
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라가 실행 중입니다. 'command+q' 키를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 화면에 출력
    cv2.imshow('B0433 Camera Stream', frame)

    # 'command+s' 키를 누르면 스냅샷 저장
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("스냅샷 저장 완료: snapshot.jpg")

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()