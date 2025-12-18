#!/usr/bin/env python3
"""
IDS U3-3991SE-C-HQ 카메라 헤드리스 캡처 프로그램 (GUI 없음)
- OpenCV 디스플레이 제거
- Tkinter GUI 제거
- 자동으로 이미지 저장
"""

import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_peak_ipl
import cv2
import numpy as np
import os
from datetime import datetime

def initialize_camera():
    """카메라 초기화"""
    ids_peak.Library.Initialize()
    
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()
    
    if device_manager.Devices().empty():
        raise RuntimeError("[❌] IDS 카메라가 발견되지 않음")
    
    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    print(f"[✔️] 카메라 연결: {device.ModelName()}")
    
    nodemap = device.RemoteDevice().NodeMaps()[0]
    
    # Gain 설정
    try:
        gain_node = nodemap.FindNode("Gain")
        if gain_node:
            gain_node.SetValue(1.0)
            print(f"[✔️] Gain 설정: {gain_node.Value():.2f}")
    except Exception as e:
        print(f"[⚠️] Gain 설정 실패: {e}")
    
    # Exposure 설정
    try:
        exposure_node = nodemap.FindNode("ExposureTime")
        if exposure_node:
            exposure_node.SetValue(30000.0)
            print(f"[✔️] Exposure 설정: {exposure_node.Value():.2f} μs")
    except Exception as e:
        print(f"[⚠️] Exposure 설정 실패: {e}")
    
    # 픽셀 포맷 설정
    try:
        pixel_format_node = nodemap.FindNode("PixelFormat")
        if pixel_format_node:
            pixel_format_node.SetCurrentEntry("RGB8")
            print(f"[✔️] PixelFormat: {pixel_format_node.CurrentEntry().SymbolicValue()}")
    except Exception as e:
        print(f"[⚠️] PixelFormat 설정 실패: {e}")
    
    # DataStream 설정
    datastreams = device.DataStreams()
    if datastreams.empty():
        device.Close()
        raise RuntimeError("[❌] DataStream이 없음")
    
    datastream = datastreams[0].OpenDataStream()
    
    # 버퍼 할당 (PayloadSize 필요)
    payload_size = nodemap.FindNode("PayloadSize").Value()
    buffer_count_max = datastream.NumBuffersAnnouncedMinRequired()
    print(f"[✔️] PayloadSize: {payload_size} bytes")
    print(f"[✔️] Buffer count: {buffer_count_max}")
    
    for i in range(buffer_count_max):
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)
    
    datastream.StartAcquisition()
    nodemap.FindNode("AcquisitionStart").Execute()
    print("[✔️] 카메라 초기화 완료\n")
    
    return device, datastream, nodemap

def center_crop(image, crop_width, crop_height):
    """이미지 중앙 크롭"""
    h, w = image.shape[:2]
    start_x = max(0, (w - crop_width) // 2)
    start_y = max(0, (h - crop_height) // 2)
    end_x = min(w, start_x + crop_width)
    end_y = min(h, start_y + crop_height)
    return image[start_y:end_y, start_x:end_x]

def capture_and_save(device, datastream, nodemap, num_frames=10, crop_width=4096, crop_height=4096):
    """헤드리스 캡처 및 저장"""
    save_dir = "saved_frames"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[▶] {num_frames}장의 이미지 캡처 시작...")
    print(f"[▶] Crop 크기: {crop_width} x {crop_height}")
    print(f"[▶] 저장 디렉토리: {save_dir}\n")
    
    try:
        for i in range(num_frames):
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
            
            # 원본 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            original_path = os.path.join(save_dir, f"frame_{timestamp}_original.png")
            cv2.imwrite(original_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 크롭 저장
            cropped = center_crop(frame, crop_width, crop_height)
            cropped_path = os.path.join(save_dir, f"frame_{timestamp}_crop.png")
            cv2.imwrite(cropped_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            print(f"[{i+1}/{num_frames}] 저장 완료:")
            print(f"  - 원본: {original_path} ({frame.shape[1]}x{frame.shape[0]})")
            print(f"  - 크롭: {cropped_path} ({cropped.shape[1]}x{cropped.shape[0]})")
            
            datastream.QueueBuffer(buffer)
    
    finally:
        print("\n[🛑] 캡처 종료 중...")
        nodemap.FindNode("AcquisitionStop").Execute()
        datastream.KillWait()
        datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        
        for buffer in datastream.AnnouncedBuffers():
            datastream.RevokeBuffer(buffer)
        
        # Device는 자동으로 닫힘 (Close 메소드 없음)
        ids_peak.Library.Close()
        print("[✔️] 카메라 종료 완료")

if __name__ == "__main__":
    device, datastream, nodemap = initialize_camera()
    capture_and_save(device, datastream, nodemap, num_frames=5, crop_width=4096, crop_height=4096)
    print("\n[✅] 모든 작업 완료!")
