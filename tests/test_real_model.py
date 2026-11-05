"""실제 PaddleOCR 모델로 확인하는 시험. T-07과 외부 통신 없는 로컬 실행."""
from __future__ import annotations

import socket

from run_t07 import run

from campusform2excel.ocr import PaddleEngine, verify_model


def test_t07_thirty_cell_contract(paddle_engine):
    report = run(paddle_engine)
    assert report["total_cells"] == 30
    assert report["missing_results"] == 0
    assert report["contract_violations"] == []
    assert report["by_kind"]["blank"]["status"] == {"empty": 10}
    assert report["passed"]


def test_model_files_match_the_lock():
    assert verify_model()


def test_model_loads_and_reads_without_external_network(monkeypatch):
    """T-N4 사전 점검: 외부 연결을 막아도 모델 로딩과 인식이 된다."""
    attempts = []
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            attempts.append(address)
            raise OSError(f"외부 연결 차단: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    report = run(PaddleEngine())
    assert report["passed"]
    assert attempts == []
