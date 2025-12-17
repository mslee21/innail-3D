import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 현재 밝기 값 가져오기
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
print(f"🎛 현재 밝기 값: {brightness}")

# 밝기를 변경할 수 있는지 테스트
new_brightness = brightness + 10
cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
updated_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)

print(f"🎚 변경된 밝기 값: {updated_brightness}")

cap.release()