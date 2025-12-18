#!/usr/bin/env python3
"""
Arducam 12MP B0433 UVC 카메라 클래스
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from camera_base import CameraBase

class CameraB0433(CameraBase):
    """Arducam 12MP B0433 UVC 카메라"""
    
    def __init__(self, save_dir: str = "saved_frames",
                 camera_index: int = 0,
                 width: int = 1920,
                 height: int = 1080,
                 fps: int = 30):
        """
        Args:
            save_dir: 이미지 저장 디렉토리
            camera_index: 카메라 인덱스 (/dev/videoX)
            width: 프레임 너비
            height: 프레임 높이
            fps: FPS
        """
        super().__init__("Arducam-B0433", save_dir)
        
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame_count = 0
        self.fps_start_time = None
    
    def initialize(self):
        """카메라 초기화"""
        if self.is_initialized:
            print(f"[⚠️] {self.name} 이미 초기화됨")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"[❌] {self.name} 카메라를 열 수 없음 (index: {self.camera_index})")
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 실제 설정 값 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.is_initialized = True
            print(f"[✔️] {self.name} 초기화 완료")
            print(f"    - 해상도: {actual_width}x{actual_height}")
            print(f"    - FPS: {actual_fps}")
            
        except Exception as e:
            print(f"[❌] {self.name} 초기화 실패: {e}")
            self.release()
            raise
    
    def start_preview(self):
        """실시간 프리뷰 시작"""
        if not self.is_initialized:
            raise RuntimeError(f"[❌] {self.name} 초기화되지 않음")
        
        window_name = f"{self.name} Live Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        self.is_running = True
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        print(f"\n[▶] {self.name} 실시간 프리뷰 시작")
        print("[키보드 단축키]")
        print("  ESC/Q: 종료")
        print("  S: 현재 프레임 저장")
        print("  FPS가 30프레임마다 출력됩니다\n")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[⚠️] 프레임을 읽을 수 없음")
                    break
                
                self.frame_count += 1
                
                # FPS 계산 및 표시
                if self.frame_count >= 30:
                    now = time.time()
                    elapsed = now - self.fps_start_time
                    current_fps = self.frame_count / elapsed
                    print(f"[FPS] {current_fps:.2f}")
                    
                    # 화면에 FPS 표시
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    self.fps_start_time = now
                    self.frame_count = 0
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("[🛑] 사용자가 종료를 요청했습니다.")
                    break
                elif key == ord('s'):  # Save
                    filename = self.generate_filename(f"{self.name}_frame")
                    save_path = self.get_save_path(filename)
                    cv2.imwrite(save_path, frame)
                    print(f"[💾] 저장 완료: {save_path} ({frame.shape[1]}x{frame.shape[0]})")
        
        finally:
            cv2.destroyAllWindows()
            self.is_running = False
    
    def capture_frame(self, filename: str = None):
        """단일 프레임 캡처"""
        if not self.is_initialized:
            raise RuntimeError(f"[❌] {self.name} 초기화되지 않음")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"[❌] {self.name} 프레임을 읽을 수 없음")
        
        if filename is None:
            filename = self.generate_filename(f"{self.name}_frame")
        
        save_path = self.get_save_path(filename)
        cv2.imwrite(save_path, frame)
        
        return save_path
    
    def stop(self):
        """카메라 정지"""
        self.is_running = False
    
    def release(self):
        """리소스 해제"""
        if not self.is_initialized:
            return
        
        try:
            if self.cap:
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            print(f"[✔️] {self.name} 리소스 해제 완료")
        
        except Exception as e:
            print(f"[⚠️] {self.name} 리소스 해제 중 오류: {e}")
        
        finally:
            self.cap = None
            self.is_initialized = False
            self.is_running = False
