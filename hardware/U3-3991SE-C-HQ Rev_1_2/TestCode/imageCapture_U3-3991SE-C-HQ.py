import cv2
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

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
    cv2.resizeWindow(window_name, crop_width, crop_height)

    os.makedirs("saved_frames", exist_ok=True)

    # Tkinter UI 구성
    root = tk.Tk()
    root.title("Crop 설정")

    # 변수 정의
    crop_w_var = tk.IntVar(value=crop_width)
    crop_h_var = tk.IntVar(value=crop_height)

    tk.Label(root, text="Crop Width:").grid(row=0, column=0)
    tk.Entry(root, textvariable=crop_w_var, width=10).grid(row=0, column=1)

    tk.Label(root, text="Crop Height:").grid(row=1, column=0)
    tk.Entry(root, textvariable=crop_h_var, width=10).grid(row=1, column=1)

    def apply_crop():
        nonlocal crop_width, crop_height
        w = crop_w_var.get()
        h = crop_h_var.get()
        crop_width = max(100, int(w))
        crop_height = max(100, int(h))
        print(f"[✔️] 새로운 crop 적용: {crop_width} x {crop_height}")
        cv2.resizeWindow(window_name, crop_width, crop_height)

    tk.Button(root, text="Crop 적용", command=apply_crop).grid(row=2, columnspan=2, pady=5)

    print("[▶] 실시간 영상 스트리밍 시작 (ESC로 종료, S 키로 저장)")

    # Tk 창 위치 설정 (선택)
    root.geometry("+100+100")

    try:
        while True:
            root.update()  # Tkinter UI 갱신
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
            info_text = f"Crop: {crop_width} x {crop_height}"
            cv2.putText(preview, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('s'):
                save_root = tk.Tk()
                save_root.withdraw()
                default_filename = datetime.now().strftime("frame_%Y%m%d_%H%M%S.png")
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png")],
                    initialdir="saved_frames",
                    initialfile=default_filename,
                    title="프레임 저장 위치 선택"
                )
                save_root.destroy()
                if save_path:
                    cv2.imwrite(save_path, frame)
                    print(f"[💾] 원본 해상도로 저장됨: {save_path}")

            datastream.QueueBuffer(buffer)

    finally:
        root.destroy()
        nodemap.FindNode("AcquisitionStop").Execute()
        datastream.KillWait()
        datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in datastream.AnnouncedBuffers():
            datastream.RevokeBuffer(buffer)
        device.Close()
        ids_peak.Library.Close()
        cv2.destroyAllWindows()
        print("[🛑] 스트리밍 종료")

if __name__ == "__main__":
    device, datastream, nodemap = initialize_camera()
    stream_preview(device, datastream, nodemap)
