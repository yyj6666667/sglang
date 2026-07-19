import importlib


class _ForwardMode:
    def __init__(self, decode):
        self._decode = decode

    def is_decode(self):
        return self._decode


class _ForwardBatch:
    def __init__(self, decode):
        self.forward_mode = _ForwardMode(decode)


def _reload(monkeypatch, enabled):
    monkeypatch.setenv("SGLANG_DECODE_PROFILE", "1" if enabled else "0")
    import sglang.srt.utils.decode_profile as decode_profile

    return importlib.reload(decode_profile)


def test_decode_profile_range_is_noop_by_default(monkeypatch):
    module = _reload(monkeypatch, enabled=False)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    with module.decode_profile_range("model", _ForwardBatch(decode=True)):
        pass


def test_decode_profile_range_marks_decode_only(monkeypatch):
    module = _reload(monkeypatch, enabled=True)
    names = []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda.nvtx,
        "range",
        lambda name: _RecordingRange(names, name),
    )

    with module.decode_profile_range("model", _ForwardBatch(decode=False)):
        pass
    with module.decode_profile_range("model", _ForwardBatch(decode=True)):
        pass

    assert names == ["sglang.decode.model"]


class _RecordingRange:
    def __init__(self, names, name):
        self.names = names
        self.name = name

    def __enter__(self):
        self.names.append(self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        return False
