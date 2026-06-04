import sys
from types import SimpleNamespace
from unittest.mock import Mock

from leann import embedding_compute
from leann.chat import _run_with_optional_posix_alarm
from leann.embedding_compute import _query_lmstudio_context_limit


def test_hf_loader_timeout_helper_runs_without_sigalrm(monkeypatch):
    monkeypatch.setitem(sys.modules, "signal", SimpleNamespace())

    assert _run_with_optional_posix_alarm(lambda: "loaded", 60, "timeout") == "loaded"


def test_hf_loader_timeout_helper_runs_when_signal_registration_is_unavailable(monkeypatch):
    fake_signal = SimpleNamespace(
        SIGALRM=14,
        alarm=Mock(side_effect=AssertionError("alarm should not be called")),
        signal=Mock(side_effect=ValueError("signal only works in main thread")),
    )
    monkeypatch.setitem(sys.modules, "signal", fake_signal)

    assert _run_with_optional_posix_alarm(lambda: "loaded", 60, "timeout") == "loaded"


def test_lmstudio_node_path_uses_platform_separator(monkeypatch):
    captured_node_path = None

    def mock_run(cmd, **kwargs):
        nonlocal captured_node_path
        if cmd == ["npm", "root", "-g"]:
            result = Mock()
            result.returncode = 0
            result.stdout = r"C:\npm\node_modules" + "\n"
            result.stderr = ""
            return result

        assert cmd[0] == "node"
        captured_node_path = kwargs["env"]["NODE_PATH"]
        result = Mock()
        result.returncode = 0
        result.stdout = '{"contextLength": 8192, "identifier": "custom-model"}'
        result.stderr = ""
        return result

    monkeypatch.setattr(embedding_compute.subprocess, "run", mock_run)
    monkeypatch.setattr(embedding_compute.os, "pathsep", ";")
    monkeypatch.setenv("NODE_PATH", r"C:\existing;D:\more")

    limit = _query_lmstudio_context_limit(model_name="custom-model", base_url="ws://localhost:1234")

    assert limit == 8192
    assert captured_node_path == r"C:\npm\node_modules;C:\existing;D:\more"
