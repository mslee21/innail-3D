import cv2
import time

# 🎥 카메라 인덱스 설정 (0, 1, 2 등으로 변경 가능)
CAMERA_INDEX = 0

# 🔍 카메라 속성 목록 (UVC에서 지원하는 기본 속성)
CAMERA_SETTINGS = {
    "Brightness": cv2.CAP_PROP_BRIGHTNESS,
    "Contrast": cv2.CAP_PROP_CONTRAST,
    "Saturation": cv2.CAP_PROP_SATURATION,
    "Hue": cv2.CAP_PROP_HUE,
    "Gain": cv2.CAP_PROP_GAIN,
    "Exposure": cv2.CAP_PROP_EXPOSURE,
    "Auto Exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "White Balance": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
    "Auto White Balance": cv2.CAP_PROP_AUTO_WB,
    "Focus": cv2.CAP_PROP_FOCUS,
    "Auto Focus": cv2.CAP_PROP_AUTOFOCUS,
}
# 📏 지원되는 해상도 목록
RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480), (320, 240)]
current_res_index = 0  # 현재 해상도 인덱스


# 🎥 카메라 열기
#cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows는 CAP_DSHOW 사용, macOS/Linux는 생략 가능
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Windows 환경에서는 CAP_DSHOW 권장
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 📢 현재 카메라 속성 확인
print("\n🎥 현재 카메라 속성:")
for name, prop in CAMERA_SETTINGS.items():
    value = cap.get(prop)
    if value != -1:  # 지원되지 않는 속성은 -1 반환
        print(f"{name}: {value}")

# 📏 기본 해상도 설정
width, height = RESOLUTIONS[current_res_index]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 🎚 속성 조정 변수 (조정값 증가/감소량)
step = 5  # 일반 속성 변경 범위
exposure_step = 0.1  # 노출 변경 범위

print("\n🎥 스트리밍 시작! (키 입력으로 속성 조정 가능)")
print("""
=== 조작 방법 ===
- 'a' / 'b' : 밝기 증가 / 감소
- 'c' / 'd' : 대비 증가 / 감소
- 'e' / 'f' : 채도 증가 / 감소
- 'g' / 'h' : 노출 증가 / 감소
- 'x'        : 자동 노출 ON/OFF
- 'y'        : 자동 초점 ON/OFF
- 'w'        : 자동 화이트 밸런스 ON/OFF
- 'r'        : 해상도 변경
- 'q'        : 종료
""")

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
        value = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, value + step)
        print(f"🔆 밝기 증가: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord("b"):  # 밝기 감소
        value = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, value - step)
        print(f"🔅 밝기 감소: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord("c"):  # 대비 증가
        value = cap.get(cv2.CAP_PROP_CONTRAST)
        cap.set(cv2.CAP_PROP_CONTRAST, value + step)
        print(f"🎚 대비 증가: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    elif key == ord("d"):  # 대비 감소
        value = cap.get(cv2.CAP_PROP_CONTRAST)
        cap.set(cv2.CAP_PROP_CONTRAST, value - step)
        print(f"🎚 대비 감소: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    elif key == ord("e"):  # 채도 증가
        value = cap.get(cv2.CAP_PROP_SATURATION)
        cap.set(cv2.CAP_PROP_SATURATION, value + step)
        print(f"🎨 채도 증가: {cap.get(cv2.CAP_PROP_SATURATION)}")
    elif key == ord("f"):  # 채도 감소
        value = cap.get(cv2.CAP_PROP_SATURATION)
        cap.set(cv2.CAP_PROP_SATURATION, value - step)
        print(f"🎨 채도 감소: {cap.get(cv2.CAP_PROP_SATURATION)}")
    elif key == ord("g"):  # 노출 증가
        value = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, value + exposure_step)
        print(f"📸 노출 증가: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("h"):  # 노출 감소
        value = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, value - exposure_step)
        print(f"📸 노출 감소: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    elif key == ord("x"):  # 자동 노출 ON/OFF
        auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        new_value = 1 if auto_exp == 0 else 0
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new_value)
        print(f"🔄 자동 노출 {'활성화' if new_value else '비활성화'}")
    elif key == ord("y"):  # 자동 초점 ON/OFF
        auto_focus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        new_value = 1 if auto_focus == 0 else 0
        cap.set(cv2.CAP_PROP_AUTOFOCUS, new_value)
        print(f"🔄 자동 초점 {'활성화' if new_value else '비활성화'}")
    elif key == ord("w"):  # 자동 화이트 밸런스 ON/OFF
        auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
        new_value = 1 if auto_wb == 0 else 0
        cap.set(cv2.CAP_PROP_AUTO_WB, new_value)
        print(f"🔄 자동 화이트 밸런스 {'활성화' if new_value else '비활성화'}")
    elif key == ord("r"):  # 해상도 변경
        current_res_index = (current_res_index + 1) % len(RESOLUTIONS)
        width, height = RESOLUTIONS[current_res_index]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"📏 해상도 변경: {width}x{height}")

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print("✅ 테스트 완료. 카메라 종료됨.")