import cv2

cap = cv2.VideoCapture(0,)  # CAP_AVFOUNDATION 대신 CAP_ANY 사용
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 모드
cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 자동 화이트 밸런스 비활성화

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 지원하는 카메라 속성 확인
print("\n🎥 현재 카메라 속성:")
properties = [
    ("Brightness", cv2.CAP_PROP_BRIGHTNESS),
    ("Contrast", cv2.CAP_PROP_CONTRAST),
    ("Saturation", cv2.CAP_PROP_SATURATION),
    ("Hue", cv2.CAP_PROP_HUE),
    ("Gain", cv2.CAP_PROP_GAIN),
    ("Exposure", cv2.CAP_PROP_EXPOSURE),
    ("Auto Exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("White Balance", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
    ("Auto White Balance", cv2.CAP_PROP_AUTO_WB),
    ("Focus", cv2.CAP_PROP_FOCUS),
    ("Auto Focus", cv2.CAP_PROP_AUTOFOCUS),
]

for name, prop in properties:
    value = cap.get(prop)
    if value == -1:
        print(f"⚠️ {name}: 지원되지 않음")
    else:
        print(f"{name}: {value}")

cap.release()