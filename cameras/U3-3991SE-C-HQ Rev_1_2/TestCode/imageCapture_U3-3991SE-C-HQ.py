import cv2
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import numpy as np
import os
from datetime import datetime

def initialize_camera():
    ids_peak.Library.Initialize()
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()

    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    datastream = device.DataStreams()[0].OpenDataStream()
    nodemap = device.RemoteDevice().NodeMaps()[0]

    nodemap.FindNode("ExposureTime").SetValue(30000.0)
    payload_size = nodemap.FindNode("PayloadSize").Value()
    buffer_count = datastream.NumBuffersAnnouncedMinRequired()

    for _ in range(buffer_count):
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)

    datastream.StartAcquisition()
    nodemap.FindNode("AcquisitionStart").Execute()

    return device, datastream, nodemap

def center_crop(image, crop_width, crop_height):
    h, w = image.shape[:2]
    start_x = max(0, w // 2 - crop_width // 2)
    start_y = max(0, h // 2 - crop_height // 2)
    end_x = min(w, start_x + crop_width)
    end_y = min(h, start_y + crop_height)
    return image[start_y:end_y, start_x:end_x]

def stream_preview(device, datastream, nodemap, crop_width=4096, crop_height=4096):
    window_name = "U3-3991SE Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 프리뷰 창 크기를 화면에 맞게 조정 (1024x1024)
    preview_window_size = 1024
    cv2.resizeWindow(window_name, preview_window_size, preview_window_size)

    os.makedirs("saved_frames", exist_ok=True)
    
    # 키보드 단축키 안내
    print("[▶] 실시간 영상 스트리밍 시작")
    print("[키보드 단축키]")
    print("  ESC/Q: 종료")
    print("  S: 현재 프레임 저장")
    print("  +/=: Crop 크기 증가 (256px)")
    print("  -/_: Crop 크기 감소 (256px)")
    print("  R: 원본 해상도로 리셋")
    print()

    try:
        while True:
            buffer = datastream.WaitForFinishedBuffer(5000)

            ipl_image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height()
            )

            converted = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
            frame = converted.get_numpy_3D()

            preview = center_crop(frame, crop_width, crop_height)
            
            # 화면에 정보 표시
            info_lines = [
                f"Crop: {crop_width} x {crop_height}",
                f"Original: {frame.shape[1]} x {frame.shape[0]}",
                "Press 'H' for help"
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(preview, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            
            # ESC 또는 Q: 종료
            if key == 27 or key == ord('q'):
                print("[🛑] 사용자가 종료를 요청했습니다.")
                break
            
            # S: 프레임 저장
            elif key == ord('s'):
                timestamp = datetime.now().strftime("frame_%Y%m%d_%H%M%S.png")
                save_path = os.path.join("saved_frames", timestamp)
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"[💾] 저장 완료: {save_path} ({frame.shape[1]}x{frame.shape[0]})")
            
            # +/=: Crop 크기 증가
            elif key == ord('+') or key == ord('='):
                crop_width = min(crop_width + 256, frame.shape[1])
                crop_height = min(crop_height + 256, frame.shape[0])
                print(f"[✔️] Crop 크기 증가: {crop_width} x {crop_height}")
            
            # -/_: Crop 크기 감소
            elif key == ord('-') or key == ord('_'):
                crop_width = max(crop_width - 256, 256)
                crop_height = max(crop_height - 256, 256)
                print(f"[✔️] Crop 크기 감소: {crop_width} x {crop_height}")
            
            # R: 리셋
            elif key == ord('r'):
                crop_width = frame.shape[1]
                crop_height = frame.shape[0]
                print(f"[✔️] 원본 해상도로 리셋: {crop_width} x {crop_height}")
            
            # H: 도움말
            elif key == ord('h'):
                print("\n[키보드 단축키]")
                print("  ESC/Q: 종료")
                print("  S: 현재 프레임 저장")
                print("  +/=: Crop 크기 증가 (256px)")
                print("  -/_: Crop 크기 감소 (256px)")
                print("  R: 원본 해상도로 리셋")
                print("  H: 도움말 표시\n")

            datastream.QueueBuffer(buffer)

    finally:
        nodemap.FindNode("AcquisitionStop").Execute()
        datastream.KillWait()
        datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in datastream.AnnouncedBuffers():
            datastream.RevokeBuffer(buffer)
        # Device는 자동으로 닫힘 (Close 메소드 없음)
        ids_peak.Library.Close()
        cv2.destroyAllWindows()
        print("[✔️] 스트리밍 종료 완료")

if __name__ == "__main__":
    device, datastream, nodemap = initialize_camera()
    stream_preview(device, datastream, nodemap)
