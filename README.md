# Innail-3D

> **Plenoptic Light Field Camera for Hand & Nail Diagnosis System**  
> 플렌옵틱 카메라를 활용한 손톱 및 손 3D 분석 진단 시스템

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

---

## 📖 목차
- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [설치 방법](#-설치-방법)
- [사용법](#-사용법)
- [프로젝트 구조](#-프로젝트-구조)
- [하드웨어](#-하드웨어)
- [개발 로드맵](#-개발-로드맵)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)

---

## 🎯 프로젝트 개요

**Innail-3D**는 Light Field 이미징 기술을 활용하여 손톱의 3D 깊이 정보를 추출하고, 이를 기반으로 손톱 및 손의 건강 상태를 분석하는 연구 개발 프로젝트입니다.

### 핵심 기술
- **Plenoptic 1.0/2.0** 재구성 알고리즘
- **Microlens Array (MLA)** 기반 광학 시스템
- **Deep Learning** 기반 Depth Estimation (Depth-Anything)
- **실시간 영상 처리** 및 3D 시각화

### 연구 목표
- 비침습적 손톱 건강 진단
- 손톱 표면 형태 및 색상 분석
- 깊이 정보를 활용한 3D 프로파일링
- AI 기반 자동 진단 시스템 구축

---

## ✨ 주요 기능

### 1. **다중 카메라 지원**
- **Arducam 12MP B0433**: 30fps 실시간 UVC 스트리밍
- **IDS U3-3991SE-C-HQ**: 4K 고해상도 산업용 카메라

### 2. **자동 캘리브레이션**
- FFT 기반 MLA pitch 자동 추정
- CLAHE 전처리 및 피크 검출
- Plenopticam 통합 보정 파이프라인

### 3. **Depth Map 생성**
- Light Field 재구성 알고리즘
- Depth-Anything 딥러닝 모델 통합
- 3D Point Cloud 시각화

### 4. **광학 시뮬레이션**
- MTF (Modulation Transfer Function) 분석
- DOF (Depth of Field) 계산
- Ray Tracing 시뮬레이션

### 5. **실시간 영상 제어**
- 밝기, 대비, 노출, 포커스 조정
- 중심 크롭 및 해상도 변경
- GUI 기반 인터랙티브 컨트롤

---

## 🔬 기술 스택

### 카메라 제어
- `OpenCV` - UVC 카메라 인터페이스
- `IDS Peak SDK` - 산업용 카메라 제어
- `V4L2` - Linux 비디오 장치 접근

### 영상 처리
- `NumPy` - 고속 배열 연산
- `OpenCV` - 영상 처리 및 컴퓨터 비전
- `PIL/Pillow` - 이미지 로딩/저장
- `SciPy` - 신호 처리 및 FFT

### Deep Learning
- `PyTorch` - 딥러닝 프레임워크
- `Depth-Anything` - Depth Estimation 모델
- `torchvision` - 영상 변환 유틸리티

### Light Field Processing
- `Plenopticam (ETRI Custom)` - 플렌옵틱 재구성 라이브러리
  - Light Field Calibration
  - View Point Extraction
  - Refocusing
  - Depth Extraction

### GUI & Visualization
- `PyQt5` - 데스크톱 애플리케이션
- `Tkinter` - 간단한 UI 컨트롤
- `Matplotlib` - 그래프 및 3D 시각화

---

## 🚀 설치 방법

### 시스템 요구사항
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python**: 3.8 이상
- **GPU**: CUDA 지원 GPU (권장, Depth-Anything 사용 시)
- **RAM**: 16GB 이상 권장

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/innail-3D.git
cd innail-3D
```

### 2. Git LFS 설정 (대용량 파일 관리)
```bash
git lfs install
git lfs pull
```

### 3. 환경 설정
```bash
bash setup.sh
```

### 4. 의존성 설치
```bash
pip install -r requirements.txt
```

### 5. IDS 카메라 드라이버 설치 (선택사항)
```bash
cd software/ids-3991/ids-software-suite-linux-64-4.95.2-debian
sudo ./ueye_4.94.2.1258_amd64.run
```

---

## 📚 사용법

### 카메라 실시간 스트리밍
```bash
# Arducam 30fps 캡처
python hardware/Aducam_12MP_B0433/B0433_testcode/ImageCapture_30fps_B0433_opencv.py

# IDS 카메라 고해상도 캡처
python hardware/U3-3991SE-C-HQ\ Rev_1_2/TestCode/imageCapture_U3-3991SE-C-HQ.py
```

### 캘리브레이션 실행
```python
python software/reconstruction/centroid_extraction/calibration_0426.py
```

### Depth Map 생성
```python
python software/depth_map/PlenoMatrix_Innail_Cert.py
```

### UVC 카메라 속성 조정
```python
python software/testcode/B0433_UVC_attribute_test.py
```

**키보드 컨트롤:**
- `↑/↓`: 밝기 조정
- `←/→`: 대비 조정
- `+/-`: 노출 조정
- `f/F`: 포커스 조정
- `s`: 프레임 저장
- `ESC`: 종료

---

## 📂 프로젝트 구조

```plaintext
innail-3D/
│
├── hardware/                         # 🔌 하드웨어 인터페이스 및 테스트 코드
│   ├── Aducam_12MP_B0433/            # Arducam 12MP 카메라 모듈
│   │   └── B0433_testcode/
│   │       ├── ImageCapture_30fps_B0433_opencv.py    # 30fps 실시간 캡처
│   │       ├── ImageCapture_GUI.py                   # PyQt5 GUI
│   │       └── optical_simulation/                   # MTF, DOF 시뮬레이션
│   │
│   └── U3-3991SE-C-HQ Rev_1_2/       # IDS 산업용 카메라
│       ├── TestCode/
│       │   └── imageCapture_U3-3991SE-C-HQ.py        # 4K 캡처 및 크롭
│       └── SW/                                        # 드라이버 및 문서
│
├── software/                         # 💻 소프트웨어 파이프라인
│   ├── depth_map/                    # Depth Map 추출
│   │   ├── PlenoMatrix_Innail_Cert.py                # 메인 파이프라인
│   │   └── Cert_image/                               # 캘리브레이션 데이터
│   │
│   ├── reconstruction/               # Light Field 재구성
│   │   ├── centroid_extraction/
│   │   │   └── calibration_0426.py                   # FFT 기반 MLA pitch 추정
│   │   │
│   │   └── my_library/
│   │       └── plenopticam_etri_09/                  # ETRI 커스텀 라이브러리
│   │           ├── lfp_calibrator/                   # 캘리브레이션
│   │           ├── lfp_aligner/                      # 이미지 정렬
│   │           ├── lfp_extractor/                    # View Point 추출
│   │           ├── lfp_refocuser/                    # 리포커싱
│   │           └── lfp_reader/                       # 이미지 로딩
│   │
│   ├── testcode/                     # 테스트 및 디버깅 코드
│   │   ├── B0433_UVC_attribute_test.py               # UVC 속성 실시간 조정
│   │   ├── resolution.py                             # 해상도 테스트
│   │   └── optical_simulation_MLA/                   # MLA 광학 분석
│   │
│   ├── opensource/                   # 외부 오픈소스
│   │   └── ArducamUVCPythonDemo/                     # Arducam 공식 데모
│   │
│   └── ids-3991/                     # IDS 드라이버
│       └── ids-software-suite-linux-64-4.95.2-debian/
│
├── data/                             # 📁 데이터 저장소
│   └── raw/                          # 원본 캡처 이미지
│       └── 250422/Num3/CaptureTest/
│
├── bundles/                          # 📦 빌드 및 배포 파일
│   └── windows/
│
├── requirements.txt                  # 의존성 패키지 목록
├── setup.sh                          # 환경 설정 스크립트
├── .gitignore                        # Git 제외 파일
├── .gitattributes                    # Git LFS 설정
└── README.md                         # 프로젝트 문서 (본 파일)
```

---

## 🔌 하드웨어

### 1. **Arducam 12MP B0433**
- **해상도**: 1920×1080 @ 30fps
- **인터페이스**: USB 3.0 UVC
- **센서**: IMX477 12.3MP
- **렌즈**: 커스텀 MLA 장착 가능
- **제어**: OpenCV VideoCapture API

### 2. **IDS U3-3991SE-C-HQ**
- **해상도**: 4096×4096 @ 10fps
- **인터페이스**: USB 3.0 (IDS Peak SDK)
- **센서**: CMOS 16MP
- **특징**: 산업용 고정밀 이미징
- **제어**: IDS Peak Python API

### 3. **Microlens Array (MLA)**
- **타입**: Hexagonal / Square Grid
- **Pitch**: ~17 pixels (자동 추정)
- **재질**: S-TIH53 (고굴절률 유리)
- **용도**: Light Field 영상 획득

---

## 🛣️ 개발 로드맵

### ✅ 완료된 기능
- [x] 다중 카메라 인터페이스 구현
- [x] 실시간 영상 캡처 및 저장
- [x] FFT 기반 자동 캘리브레이션
- [x] Plenoptic 재구성 파이프라인
- [x] Depth-Anything 모델 통합
- [x] UVC 속성 실시간 제어
- [x] 광학 시뮬레이션 (MTF, DOF)

### 🚧 진행 중
- [ ] 손톱 영역 자동 검출 (Segmentation)
- [ ] 특징 추출 알고리즘 (Texture, Color, Shape)
- [ ] 데이터셋 구축 및 라벨링

### 📅 향후 계획
- [ ] 딥러닝 진단 모델 개발
- [ ] 통합 GUI 애플리케이션
- [ ] REST API 서버 구축
- [ ] 모바일 앱 연동
- [ ] 임상 데이터 검증

---

## 🤝 기여하기

프로젝트에 기여하고 싶으신 분들은 아래 절차를 따라주세요:

1. **Fork** 이 저장소
2. **Feature Branch** 생성 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항 (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to Branch (`git push origin feature/AmazingFeature`)
5. **Pull Request** 생성

### 코딩 컨벤션
- Python PEP 8 스타일 가이드 준수
- 함수 및 클래스에 docstring 작성
- Type hints 사용 권장

---

## 📄 라이선스

이 프로젝트는 **MIT License**를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👥 팀 및 연락처

### 프로젝트 팀
- **24BK1300** 연구팀
- **ETRI (한국전자통신연구원)** 협력

### 문의
- 📧 Email: [이메일 주소]
- 🌐 Website: [프로젝트 웹사이트]
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/innail-3D/issues)

---

## 🙏 감사의 글

- **Plenopticam** - Stanford Computational Imaging Lab
- **Depth-Anything** - TikTok / ByteDance AI Lab
- **Arducam** - 카메라 하드웨어 지원
- **IDS Imaging** - 산업용 카메라 SDK

---

## 📚 참고 문헌

1. Adelson, E. H., & Wang, J. Y. (1992). "Single lens stereo with a plenoptic camera"
2. Ng, R. (2006). "Digital light field photography" (Stanford PhD Thesis)
3. Yang, L. et al. (2024). "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"

---

## 📊 프로젝트 상태

**개발 단계**: 연구 프로토타입 (Research Prototype)  
**버전**: 0.1.0 (Alpha)  
**최종 업데이트**: 2025년 12월

---

<p align="center">
  <i>Made with ❤️ for advancing non-invasive medical diagnostics</i>
</p>
