#!/usr/bin/env python3
"""
카메라 베이스 클래스
모든 카메라 클래스는 이 베이스 클래스를 상속받아 구현
"""

from abc import ABC, abstractmethod
import os
from datetime import datetime

class CameraBase(ABC):
    """카메라 베이스 클래스"""
    
    def __init__(self, name: str, save_dir: str = "saved_frames"):
        """
        Args:
            name: 카메라 이름
            save_dir: 이미지 저장 디렉토리
        """
        self.name = name
        self.save_dir = save_dir
        self.is_initialized = False
        self.is_running = False
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
    
    @abstractmethod
    def initialize(self):
        """카메라 초기화"""
        pass
    
    @abstractmethod
    def start_preview(self):
        """프리뷰 시작"""
        pass
    
    @abstractmethod
    def capture_frame(self, filename: str = None):
        """
        단일 프레임 캡처
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
        
        Returns:
            str: 저장된 파일 경로
        """
        pass
    
    @abstractmethod
    def stop(self):
        """카메라 정지"""
        pass
    
    @abstractmethod
    def release(self):
        """리소스 해제"""
        pass
    
    def generate_filename(self, prefix: str = "frame", ext: str = ".png") -> str:
        """
        타임스탬프 기반 파일명 생성
        
        Args:
            prefix: 파일명 접두사
            ext: 파일 확장자
        
        Returns:
            str: 생성된 파일명
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}_{timestamp}{ext}"
    
    def get_save_path(self, filename: str) -> str:
        """
        전체 저장 경로 반환
        
        Args:
            filename: 파일명
        
        Returns:
            str: 전체 경로
        """
        return os.path.join(self.save_dir, filename)
    
    def __enter__(self):
        """Context manager 진입"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.release()
        return False
    
    def __del__(self):
        """소멸자"""
        if self.is_initialized:
            try:
                self.release()
            except:
                pass
