# Target Speaker Extraction Prototype

여러 화자가 섞인 오디오에서 특정 타겟 화자의 음성만 분리/강조하는 프로젝트입니다.

## Quickstart (Windows)

```powershell
# 1. 가상환경 생성 및 활성화
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. 패키지 설치
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. 실행
python run_all.py --mode default --num_speakers 3
```

---

## 실행 모드

`run_all.py`는 3가지 모드를 지원합니다:

### 1. Default 모드 (랜덤 화자)

LibriSpeech 데이터셋에서 랜덤으로 화자를 선택하여 혼합 후 분리합니다.

```powershell
python run_all.py --mode default --num_speakers 3
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--num_speakers` | 혼합할 화자 수 | 3 |

---

### 2. Gender 모드 (성별 지정)

특정 성별(남성/여성)의 화자만 선택하여 혼합 후 분리합니다.  
동성 화자 분리 성능 테스트에 유용합니다.

```powershell
# 남성 화자 3명
python run_all.py --mode gender --gender M --num_speakers 3

# 여성 화자 3명
python run_all.py --mode gender --gender F --num_speakers 3
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gender` | 화자 성별 (M: 남성, F: 여성) | M |
| `--num_speakers` | 혼합할 화자 수 | 3 |

---

### 3. Custom 모드 (사용자 파일)

직접 녹음한 파일을 사용하여 분리합니다.  
다양한 오디오 형식을 지원합니다 (.wav, .m4a, .mp3, .flac 등).

```powershell
python run_all.py --mode custom --mixture "혼합음성.m4a" --enroll "타겟화자.m4a"
```

| 옵션 | 설명 | 필수 |
|------|------|------|
| `--mixture` | 혼합 음성 파일 경로 | ✅ |
| `--enroll` | 타겟 화자 등록 음성 파일 경로 | ✅ |

**참고**: .m4a, .mp3 형식 사용 시 `ffmpeg` 설치가 필요합니다.

```powershell
# ffmpeg 설치 (Windows)
winget install ffmpeg
```

---

## 공통 옵션

| 옵션 | 설명 |
|------|------|
| `--no-clean` | 이전 결과물 삭제 건너뛰기 |
| `--skip-data` | 데이터 준비 단계 건너뛰기 (이미 mixture.wav가 있는 경우) |

---

## 결과 파일

실행 완료 후 생성되는 주요 파일들:

| 파일 | 설명 |
|------|------|
| `target_emphasized.wav` | 기본 분리 결과 |
| `target_boosted_final.wav` | 비타겟 화자 억제 결과 |
| `target_emphasized_post.wav` | Hybrid 후처리 결과 |
| `target_final_clean.wav` | **최종 클린업 결과 (추천)** ⭐ |

---

## 파이프라인 구조

```
1. 데이터 준비 (혼합 음성 생성)
   ↓
2. SepFormer 분리 (음원 분리)
   ↓
3. 타겟 화자 선택 (ECAPA + Resemblyzer 임베딩)
   ↓
4~8. 후처리 단계들 (마스크 리파인, 부스트, 억제 등)
   ↓
9. Hybrid 후처리
   ↓
10. 최종 클린업 (노이즈 제거)
```

---

## 기술 스택

- **SpeechBrain**: 음성 처리 프레임워크
- **SepFormer**: 음성 분리 딥러닝 모델
- **ECAPA-TDNN**: 화자 임베딩/인식
- **LibriSpeech**: 영어 음성 데이터셋
