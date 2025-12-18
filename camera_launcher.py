#!/usr/bin/env python3
"""
innail-3D 카메라 통합 런처
- U3-3991SE (IDS Peak) 카메라
- B0433 (Arducam 12MP UVC) 카메라
"""

import os
import sys

# cameras 디렉토리를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cameras'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cameras', 'U3-3991SE-C-HQ Rev_1_2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cameras', 'Aducam_12MP_B0433'))

from camera_u3_3991se import CameraU3_3991SE
from camera_b0433 import CameraB0433

def print_menu():
    """메뉴 출력"""
    print("\n" + "="*60)
    print("  innail-3D Camera Launcher")
    print("="*60)
    print()
    print("  [1] U3-3991SE (IDS Peak) - 4504x4504 고해상도")
    print("  [2] B0433 (Arducam 12MP) - 1920x1080 30fps")
    print("  [Q] 종료")
    print()
    print("="*60)

def launch_camera(camera_class, *args, **kwargs):
    """
    카메라 실행 (재사용 가능)
    
    Args:
        camera_class: 카메라 클래스
        *args, **kwargs: 카메라 초기화 인자
    """
    camera = None
    try:
        # Context manager 사용으로 자동 초기화/해제
        with camera_class(*args, **kwargs) as camera:
            camera.start_preview()
    except KeyboardInterrupt:
        print("\n[🛑] 사용자가 종료했습니다.")
    except Exception as e:
        print(f"\n[❌] 카메라 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 명시적 리소스 해제
        if camera and camera.is_initialized:
            camera.release()

def main():
    """메인 함수"""
    # 작업 디렉토리를 스크립트 위치로 변경
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 카메라 설정
    cameras = {
        '1': (CameraU3_3991SE, {
            'save_dir': 'saved_frames/u3_3991se',
            'exposure_time': 30000.0,
            'gain': 1.0,
            'default_crop': 4096
        }),
        '2': (CameraB0433, {
            'save_dir': 'saved_frames/b0433',
            'camera_index': 0,
            'width': 1920,
            'height': 1080,
            'fps': 30
        })
    }
    
    while True:
        print_menu()
        choice = input("선택하세요 (1/2/Q): ").strip().upper()
        
        if choice in cameras:
            camera_class, kwargs = cameras[choice]
            launch_camera(camera_class, **kwargs)
        elif choice == 'Q':
            print("\n[✔️] 프로그램을 종료합니다.")
            break
        else:
            print("\n[⚠️] 잘못된 선택입니다. 1, 2, 또는 Q를 입력하세요.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[✔️] 프로그램을 종료합니다.")
        sys.exit(0)
