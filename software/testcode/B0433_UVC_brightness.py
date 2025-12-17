import cv2

# 🎥 카메라 인덱스
CAMERA_INDEX = 0

# 🎚 변경 값
STEP = 5  # 밝기 증가/감소 정도
EXPOSURE_STEP = 0.1  # 노출 증가/감소 정도

# 🎥 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Windows는 CAP_DSHOW 사용, macOS/Linux는 생략 가능

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 🔍 기본 속성 확인
print("\n🎥 현재 카메라 속성:")
print(f"🔆 밝기(Brightness): {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"📸 노출(Exposure): {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"🟢 자동 노출(Auto Exposure): {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

# 🛠 자동 노출을 비활성화하여 밝기 조절 가능하도록 설정
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 0: 수동 모드, 1: 자동 모드

# 🎥 실시간 영상 스트리밍
print("\n=== 조작 방법 ===")
print("- 'a' / 'b' : 밝기 증가 / 감소")
print("- 'c' / 'd' : 노출 증가 / 감소")
print("- 'x'        : 자동 노출 ON/OFF")
print("- 'q'        : 종료\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("UVC Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # 종료
        break
    elif key == ord("a"):  # 밝기 증가
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) + STEP)
        print(f"🔆 밝기 증가: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord("b"):  # 밝기 감소
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) - STEP)
        print(f"🔅 밝기 감소: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord("c"):  # 노출 증가
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + EXPOSURE_STEP)
        print(f"📸 노출 증가: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("d"):  # 노출 감소
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - EXPOSURE_STEP)
        print(f"📸 노출 감소: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("x"):  # 자동 노출 ON/OFF
        auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        new_value = 1 if auto_exp == 0 else 0
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new_value)
        print(f"🔄 자동 노출 {'활성화' if new_value else '비활성화'}")

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print("✅ 테스트 완료. 카메라 종료됨.")