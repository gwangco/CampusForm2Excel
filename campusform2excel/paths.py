"""프로젝트 경로와 OCR 모델 저장 위치."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
TEMPLATE_DIR = CONFIG_DIR / "templates"
OCR_SETTINGS_PATH = CONFIG_DIR / "ocr_settings.json"
MODEL_LOCK_PATH = CONFIG_DIR / "model_lock.json"
FORMS_DIR = ROOT / "forms"
JOBS_DIR = Path(os.environ.get("CF2E_JOBS_DIR", ROOT / "data" / "jobs"))


def model_home() -> Path:
    """모델을 둘 ASCII 전용 경로를 돌려준다.

    Paddle의 C++ 로더는 Windows에서 한글이 포함된 경로의 모델 파일을 열지 못한다.
    프로젝트 폴더가 한글 경로이므로 모델은 프로젝트 밖의 ASCII 경로에 둔다.
    사용자 이름이 한글인 PC는 LOCALAPPDATA도 쓸 수 없으므로 PROGRAMDATA로 넘어간다.
    """
    override = os.environ.get("CF2E_MODEL_HOME")
    if override:
        if not override.isascii():
            raise RuntimeError(f"CF2E_MODEL_HOME은 영문 경로여야 합니다: {override}")
        return Path(override)

    candidates = [os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMDATA"), str(Path.home() / ".cache")]
    for base in candidates:
        if base and base.isascii():
            return Path(base) / "CampusForm2Excel"
    raise RuntimeError("영문 경로의 모델 폴더를 찾지 못했습니다. CF2E_MODEL_HOME을 지정해 주세요.")
