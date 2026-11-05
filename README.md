# CampusForm2Excel

교내 수기 양식 3종(출석표, 장비 대여 장부, 실습 평가표)의 사진을 검수 가능한 표로 변환하는 로컬 앱입니다. 유료 API와 GPU 없이 CPU에서 실행하며, 사진은 PC 밖으로 전송되지 않습니다.

기준 문서는 요구명세서 v1.1입니다. 현재 진행 상황은 [11월 1주 완료 보고](docs/11월1주_완료보고.md)에서 확인합니다.

## 설치

Windows, Python 3.12 기준입니다. 프로젝트 폴더에서 실행합니다.

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\setup_model.py
```

마지막 명령은 고정된 버전의 OCR 모델(약 14MB)을 내려받고 SHA-256을 검증합니다. 인터넷은 이때만 필요합니다.

## 실행

```bat
.venv\Scripts\streamlit run app.py
```

브라우저에서 <http://localhost:8501>을 엽니다.

1. **1단계**: 양식을 고르고 사진을 올린 뒤 `보정·셀 정합 시작`을 누릅니다. 양식을 잘못 골랐으면 안내가 나오고, 양식을 바꿔 같은 사진으로 다시 정합할 수 있습니다.
2. **2단계**: `셀 인식 시작`을 누르면 셀별 인식값·신뢰도·상태를 보여 줍니다. 셀을 고르면 원본과 대조할 수 있습니다.

작성용 빈 양식은 [forms/빈양식_3종_인쇄용.pdf](forms/빈양식_3종_인쇄용.pdf)입니다.

## 시험

```bat
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\run_t07.py
```

`run_t07.py`는 T-07 결과를 `docs/시험기록`에 기록합니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `app.py` | Streamlit 화면 |
| `campusform2excel/templates.py` | 양식 설정 등록·조회·검증 (FR-01, 02, 05) |
| `campusform2excel/intake.py` | 업로드 검사 (FR-03) |
| `campusform2excel/preprocess.py` | 크기·노이즈 보정 (FR-04) |
| `campusform2excel/alignment.py` | 표 검출, 기울기 보정, 셀 정합 (FR-04, 06) |
| `campusform2excel/ocr.py` | PaddleOCR 연동, `recognize_cells` (FR-07) |
| `campusform2excel/pipeline.py` | 작업 단위 처리 흐름과 결과 저장 |
| `config/templates/` | 양식 3종 설정 |
| `config/ocr_settings.json` | OCR 실행 설정 |
| `config/model_lock.json` | 모델 리비전·해시·라이브러리 버전 |
| `forms/` | 인쇄용 빈 양식 |
| `scripts/` | 모델 확보, 빈 양식 생성, T-07 실행 |
| `tests/` | 자동 시험과 합성 시험 사진 |
| `docs/` | 완료 보고, 시험 기록 |
| `data/` | 실행 중 만들어지는 작업 폴더 (저장소 제외) |

## 저장소에 올리지 않는 것

이 저장소는 비공개이지만 다음 항목은 올리지 않습니다.

- 실제 수집 사진, 학생 명단, 실제 이름과 학번. 수집 사진은 `data/` 아래에 둡니다.
- 비밀번호, API 키, 토큰, `.env` 파일
- 가상환경과 OCR 모델 파일. 모델은 `scripts/setup_model.py`로 받습니다.

## 알아 둘 점

- **모델 위치**: Paddle은 Windows의 한글 경로에서 모델을 읽지 못합니다. 그래서 모델은 프로젝트 안이 아니라 `%LOCALAPPDATA%\CampusForm2Excel\models`에 둡니다. 사용자 이름이 한글이면 `%PROGRAMDATA%`를 쓰고, 둘 다 안 되면 `CF2E_MODEL_HOME`에 영문 경로를 지정합니다.
- **OpenCV**: `opencv-python-headless`를 함께 설치하면 `cv2`가 충돌합니다. `requirements.txt`의 `opencv-contrib-python`만 사용합니다.
- **이미지 입출력**: `cv2.imread`·`cv2.imwrite`는 한글 경로에서 실패하므로 `campusform2excel/imaging.py`의 함수를 사용합니다.
- **합성 시험 사진**: `tests/fixtures`의 사진은 글꼴로 만든 시험용입니다. 손글씨가 아니며 개발·최종·외부 데이터에 포함하지 않습니다.
- **양식 변경**: 설정을 바꿀 때는 파일을 덮어쓰지 않고 버전을 올립니다.
