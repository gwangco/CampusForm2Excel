"""한국어 OCR 모델을 고정된 버전으로 내려받고 해시를 검증한다.

    python scripts/setup_model.py          # 잠금 파일 기준으로 내려받고 검증
    python scripts/setup_model.py --lock   # 잠금 파일을 새로 작성 (모델을 올릴 때만 사용)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campusform2excel.paths import MODEL_LOCK_PATH, model_home  # noqa: E402

MODEL_NAME = "korean_PP-OCRv5_mobile_rec"
REPO_ID = f"PaddlePaddle/{MODEL_NAME}"
REVISION = "c02ecaf1f22bfd1c618cce154fd19185b47e663a"
MODEL_FILES = ["inference.json", "inference.pdiparams", "inference.yml", "config.json", "README.md"]
PACKAGES = ["paddlepaddle", "paddleocr", "paddlex", "streamlit", "opencv-contrib-python", "numpy", "pandas", "openpyxl"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(lock: dict) -> Path:
    from huggingface_hub import snapshot_download

    target = model_home() / "models" / lock["model_name"]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=lock["repo_id"],
        revision=lock["revision"],
        allow_patterns=list(lock["files"]),
        local_dir=target,
    )
    # 내려받기 도구의 캐시는 모델 실행에 쓰이지 않는다.
    cache = target / ".cache"
    if cache.exists():
        import shutil

        shutil.rmtree(cache)
    return target


def verify(lock: dict, target: Path) -> list[str]:
    problems = []
    for name, expected in lock["files"].items():
        path = target / name
        if not path.exists():
            problems.append(f"파일 없음: {name}")
        elif sha256(path) != expected["sha256"]:
            problems.append(f"해시 불일치: {name}")
    return problems


def write_lock() -> dict:
    lock = {
        "model_name": MODEL_NAME,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source": f"https://huggingface.co/{REPO_ID}",
        "license": "Apache-2.0",
        "locked_on": date.today().isoformat(),
        "files": {name: {} for name in MODEL_FILES},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in PACKAGES}},
    }
    target = download(lock)
    for name in MODEL_FILES:
        path = target / name
        lock["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    MODEL_LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"잠금 파일 작성: {MODEL_LOCK_PATH}")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true", help="잠금 파일을 새로 작성한다")
    args = parser.parse_args()

    if args.lock:
        lock = write_lock()
    else:
        lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))

    target = download(lock)
    problems = verify(lock, target)
    if problems:
        print("모델 검증 실패:", *problems, sep="\n  ")
        return 1
    print(f"모델 준비 완료: {target}")
    print(f"가중치 SHA-256: {lock['files']['inference.pdiparams']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
