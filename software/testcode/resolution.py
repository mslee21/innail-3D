import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 자동 노출을 수동으로 변경
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25: 수동 모드 (일부 카메라는 0으로 설정해야 함)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 자동 화이트 밸런스 비활성화
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 노출값 수동 설정
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # 밝기 조정

# 다시 속성 확인
print(f"🛠 변경된 노출 값: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"🛠 변경된 밝기 값: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")

cap.release()