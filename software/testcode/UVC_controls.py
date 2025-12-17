import cv2

def list_ports():
    is_working = True
    dev_port = 0
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Camera",dev_port,"not found")
        else:
            is_working = False
            print("Camera",dev_port,"found")
        dev_port += 1
    return dev_port-1

camera_index = list_ports()
cap = cv2.VideoCapture(camera_index)

if cap.isOpened():
    # 1. Automatic exposure control, Linux OS: AUTO=0, Manual=1; Windows OS: AUTO=1, Manual=0
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    # 2. Set Exposure Value:  Experiment with these values!
    exposure_value = -10  # Example: Adjust as needed. Exposure range from -13 ~ -1(int)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    print(f"Exposure set to: {exposure_value}")

    # 3. Set Gain (Optional):  Gain can also affect brightness
    gain_value = 800 # Example: Adjust as needed.
    cap.set(cv2.CAP_PROP_GAIN, gain_value)
    print(f"Gain set to: {gain_value}")

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL) # Create a window to display the video

    while(True):
        ret, frame = cap.read()
        if not ret: # Check if frame reading was successful
            print("Error reading frame. Check camera connection or video source.")
            break
        cv2.imshow('Video', frame) # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Failed to open camera.")