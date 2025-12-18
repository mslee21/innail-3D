#!/usr/bin/env python3
"""
IDS Peak U3-3991SE-C-HQ 카메라 클래스
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import numpy as np
from camera_base import CameraBase

class CameraU3_3991SE(CameraBase):
    """IDS Peak U3-3991SE-C-HQ 카메라"""
    
    def __init__(self, save_dir: str = "saved_frames", 
                 exposure_time: float = 30000.0,
                 gain: float = 1.0,
                 default_crop: int = 4096):
        """
        Args:
            save_dir: 이미지 저장 디렉토리
            exposure_time: 노출 시간 (μs)
            gain: 게인 값
            default_crop: 기본 크롭 크기
        """
        super().__init__("U3-3991SE-C-HQ", save_dir)
        
        self.exposure_time = exposure_time
        self.gain = gain
        self.crop_width = default_crop
        self.crop_height = default_crop
        self.preview_window_size = 1024
        
        self.device = None
        self.datastream = None
        self.nodemap = None
        self.original_width = 0
        self.original_height = 0
    
    def initialize(self):
        """카메라 초기화"""
        if self.is_initialized:
            print(f"[⚠️] {self.name} 이미 초기화됨")
            return
        
        try:
            ids_peak.Library.Initialize()
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            if device_manager.Devices().empty():
                raise RuntimeError(f"[❌] {self.name} 카메라를 찾을 수 없음")
            
            self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            self.datastream = self.device.DataStreams()[0].OpenDataStream()
            self.nodemap = self.device.RemoteDevice().NodeMaps()[0]
            
            # 카메라 설정
            self._configure_camera()
            
            # 버퍼 설정
            payload_size = self.nodemap.FindNode("PayloadSize").Value()
            buffer_count = self.datastream.NumBuffersAnnouncedMinRequired()
            
            for _ in range(buffer_count):
                buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
                self.datastream.QueueBuffer(buffer)
            
            self.datastream.StartAcquisition()
            self.nodemap.FindNode("AcquisitionStart").Execute()
            
            self.is_initialized = True
            print(f"[✔️] {self.name} 초기화 완료")
            print(f"    - 해상도: {self.original_width}x{self.original_height}")
            print(f"    - Exposure: {self.exposure_time:.2f} μs")
            print(f"    - Gain: {self.gain:.2f}")
            
        except Exception as e:
            print(f"[❌] {self.name} 초기화 실패: {e}")
            self.release()
            raise
    
    def _configure_camera(self):
        """카메라 파라미터 설정"""
        # Exposure 설정
        try:
            exposure_node = self.nodemap.FindNode("ExposureTime")
            if exposure_node:
                exposure_node.SetValue(self.exposure_time)
                print(f"[✔️] Exposure: {exposure_node.Value():.2f} μs")
        except Exception as e:
            print(f"[⚠️] Exposure 설정 실패: {e}")
        
        # Gain 설정
        try:
            gain_node = self.nodemap.FindNode("Gain")
            if gain_node:
                gain_node.SetValue(self.gain)
                print(f"[✔️] Gain: {gain_node.Value():.2f}")
        except Exception as e:
            print(f"[⚠️] Gain 설정 실패: {e}")
        
        # PixelFormat 설정
        try:
            pixel_format_node = self.nodemap.FindNode("PixelFormat")
            if pixel_format_node:
                pixel_format_node.SetCurrentEntry("RGB8")
                print(f"[✔️] PixelFormat: {pixel_format_node.CurrentEntry().SymbolicValue()}")
        except Exception as e:
            print(f"[⚠️] PixelFormat 설정 실패: {e}")
        
        # 해상도 저장
        try:
            width_node = self.nodemap.FindNode("Width")
            height_node = self.nodemap.FindNode("Height")
            if width_node and height_node:
                self.original_width = width_node.Value()
                self.original_height = height_node.Value()
        except:
            pass
    
    def _center_crop(self, image, crop_width, crop_height):
        """이미지 중앙 크롭"""
        h, w = image.shape[:2]
        start_x = max(0, w // 2 - crop_width // 2)
        start_y = max(0, h // 2 - crop_height // 2)
        end_x = min(w, start_x + crop_width)
        end_y = min(h, start_y + crop_height)
        return image[start_y:end_y, start_x:end_x]
    
    def start_preview(self):
        """실시간 프리뷰 시작"""
        if not self.is_initialized:
            raise RuntimeError(f"[❌] {self.name} 초기화되지 않음")
        
        window_name = f"{self.name} Live Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.preview_window_size, self.preview_window_size)
        
        self.is_running = True
        
        print(f"\n[▶] {self.name} 실시간 프리뷰 시작")
        print("[키보드 단축키]")
        print("  ESC/Q: 종료")
        print("  S: 현재 프레임 저장")
        print("  +/=: Crop 크기 증가 (256px)")
        print("  -/_: Crop 크기 감소 (256px)")
        print("  R: 원본 해상도로 리셋")
        print("  H: 도움말 표시\n")
        
        try:
            while self.is_running:
                buffer = self.datastream.WaitForFinishedBuffer(5000)
                
                ipl_image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
                    buffer.PixelFormat(),
                    buffer.BasePtr(),
                    buffer.Size(),
                    buffer.Width(),
                    buffer.Height()
                )
                
                converted = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
                frame = converted.get_numpy_3D()
                
                preview = self._center_crop(frame, self.crop_width, self.crop_height)
                
                # 화면에 정보 표시
                info_lines = [
                    f"Crop: {self.crop_width} x {self.crop_height}",
                    f"Original: {frame.shape[1]} x {frame.shape[0]}",
                    "Press 'H' for help"
                ]
                for i, line in enumerate(info_lines):
                    cv2.putText(preview, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, preview)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("[🛑] 사용자가 종료를 요청했습니다.")
                    break
                elif key == ord('s'):  # Save
                    filename = self.generate_filename(f"{self.name}_frame")
                    save_path = self.get_save_path(filename)
                    cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"[💾] 저장 완료: {save_path} ({frame.shape[1]}x{frame.shape[0]})")
                elif key == ord('+') or key == ord('='):  # Increase crop
                    self.crop_width = min(self.crop_width + 256, frame.shape[1])
                    self.crop_height = min(self.crop_height + 256, frame.shape[0])
                    print(f"[✔️] Crop 크기 증가: {self.crop_width} x {self.crop_height}")
                elif key == ord('-') or key == ord('_'):  # Decrease crop
                    self.crop_width = max(self.crop_width - 256, 256)
                    self.crop_height = max(self.crop_height - 256, 256)
                    print(f"[✔️] Crop 크기 감소: {self.crop_width} x {self.crop_height}")
                elif key == ord('r'):  # Reset
                    self.crop_width = frame.shape[1]
                    self.crop_height = frame.shape[0]
                    print(f"[✔️] 원본 해상도로 리셋: {self.crop_width} x {self.crop_height}")
                elif key == ord('h'):  # Help
                    print("\n[키보드 단축키]")
                    print("  ESC/Q: 종료")
                    print("  S: 현재 프레임 저장")
                    print("  +/=: Crop 크기 증가 (256px)")
                    print("  -/_: Crop 크기 감소 (256px)")
                    print("  R: 원본 해상도로 리셋")
                    print("  H: 도움말 표시\n")
                
                self.datastream.QueueBuffer(buffer)
        
        finally:
            cv2.destroyAllWindows()
            self.is_running = False
    
    def capture_frame(self, filename: str = None):
        """단일 프레임 캡처"""
        if not self.is_initialized:
            raise RuntimeError(f"[❌] {self.name} 초기화되지 않음")
        
        buffer = self.datastream.WaitForFinishedBuffer(5000)
        
        ipl_image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height()
        )
        
        converted = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
        frame = converted.get_numpy_3D()
        
        if filename is None:
            filename = self.generate_filename(f"{self.name}_frame")
        
        save_path = self.get_save_path(filename)
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        self.datastream.QueueBuffer(buffer)
        
        return save_path
    
    def stop(self):
        """카메라 정지"""
        self.is_running = False
    
    def release(self):
        """리소스 해제"""
        if not self.is_initialized:
            return
        
        try:
            if self.nodemap:
                self.nodemap.FindNode("AcquisitionStop").Execute()
            
            if self.datastream:
                self.datastream.KillWait()
                self.datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                
                for buffer in self.datastream.AnnouncedBuffers():
                    self.datastream.RevokeBuffer(buffer)
            
            ids_peak.Library.Close()
            cv2.destroyAllWindows()
            
            print(f"[✔️] {self.name} 리소스 해제 완료")
        
        except Exception as e:
            print(f"[⚠️] {self.name} 리소스 해제 중 오류: {e}")
        
        finally:
            self.device = None
            self.datastream = None
            self.nodemap = None
            self.is_initialized = False
            self.is_running = False
