from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

# 시험이 실제 작업 폴더를 건드리지 않도록 패키지를 읽기 전에 임시 폴더로 돌린다.
_jobs_dir = tempfile.TemporaryDirectory(prefix="cf2e-test-jobs-")
os.environ["CF2E_JOBS_DIR"] = _jobs_dir.name

from campusform2excel.forms import render_blank_form  # noqa: E402
from campusform2excel.templates import list_templates  # noqa: E402


class FakeEngine:
    """정해진 응답을 돌려주는 시험용 엔진. 응답이 예외이면 그대로 던진다."""

    model_name = "fake-model"
    model_sha256 = "fake-sha256"

    def __init__(self, responses=None, default=("가나다", 0.95)):
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[int] = []

    def predict(self, images):
        self.calls.append(len(images))
        outputs = []
        for _ in images:
            response = self.responses.pop(0) if self.responses else self.default
            if isinstance(response, Exception):
                raise response
            outputs.append(response)
        return outputs


@pytest.fixture(scope="session")
def templates():
    return list_templates()


@pytest.fixture(scope="session")
def blank_pages(templates):
    """양식별 빈 양식 BGR 이미지."""
    return {t.template_id: np.asarray(render_blank_form(t))[:, :, ::-1].copy() for t in templates}


@pytest.fixture(scope="session")
def paddle_engine():
    from campusform2excel.ocr import PaddleEngine

    return PaddleEngine()


@pytest.fixture
def fake_engine():
    return FakeEngine()
