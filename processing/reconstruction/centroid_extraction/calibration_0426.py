import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fftshift, fft2
my_library = "/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/reconstruction/my_library/"
sys.path.append(my_library)
# Plenopticam 불러오기
import plenopticam_etri_09 as pcam


# =========================================
# 1. MLA pitch 자동 추정 함수
# =========================================
def estimate_mla_pitch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image {image_path} could not be loaded.")

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 중앙부 crop
    h, w = enhanced.shape
    crop = enhanced[h // 2 - 200:h // 2 + 200, w // 2 - 200:w // 2 + 200]

    # FFT 변환
    fft_img = fft2(crop)
    fft_shift = fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_log = np.log1p(magnitude_spectrum)

    # 1D Profile 생성 (가운데 수평선)
    center = np.array(magnitude_log.shape) // 2
    profile_line = magnitude_log[center[0], :]

    # 피크 검출
    peaks, _ = find_peaks(profile_line, distance=5, prominence=0.05)

    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        avg_distance = np.mean(peak_distances)
    else:
        avg_distance = None

    # 시각화 (디버깅용)
    plt.figure(figsize=(10, 4))
    plt.plot(profile_line)
    plt.plot(peaks, profile_line[peaks], "x")
    plt.title("FFT 1D Profile - MLA peak detection")
    plt.grid(True)
    plt.show()

    if avg_distance:
        print(f"✅ Estimated MLA pitch: {avg_distance:.2f} pixels")
    else:
        print("⚠️ Peak detection failed. Check input image.")

    return avg_distance


# =========================================
# 2. Plenopticam 세팅 및 Calibration 실행 함수
# =========================================
def run_plenopticam_with_pitch(image_path, estimated_pitch):
    # 1. Config 설정
    cfg = pcam.cfg.PlenopticamConfig()
    cfg.default_values()

    # Calibration 이미지 경로 설정
    cfg.params[cfg.cal_path] = image_path

    # 자동 추정된 MLA pitch 적용
    estimated_pitch = 17
    cfg.params[cfg.ptc_leng] = estimated_pitch
    # cfg.calibs[cfg.grid_type] = 'hex'

    # Calibration 옵션 활성화
    cfg.params[cfg.opt_cali] = True
    cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[0]  # 'area' 방법 추천

    # 2. 상태 객체 생성
    sta = pcam.misc.PlenopticamStatus()

    # 3. 이미지 불러오기
    wht_img = pcam.misc.load_img_file(cfg.params[cfg.cal_path])
    print(f"wht_img loaded: shape={wht_img.shape}")

    # 4. Calibration 실행
    cal_obj = pcam.lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
    ret = cal_obj.main()

    if ret:
        print("✅ Calibration finished successfully!")
    else:
        print("⚠️ Calibration failed.")

    return cfg


# =========================================
# 3. 메인 실행
# =========================================
if __name__ == "__main__":
    # (1) Calibration 이미지 경로 설정
    image_path = "/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/reconstruction/centroid_extraction/test_data/a-1.bmp"  # ❗ 수정하세요

    # (2) MLA pitch 추정
    estimated_pitch = estimate_mla_pitch(image_path)

    # (3) Plenopticam 실행
    if estimated_pitch:
        cfg = run_plenopticam_with_pitch(image_path, estimated_pitch)