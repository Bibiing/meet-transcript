from rttranscriber.pysoundio_runtime import _DLL_PATH, ffi, lib


def test_runtime_loads() -> None:
    assert _DLL_PATH.exists()
    version = ffi.string(lib.soundio_version_string()).decode()
    assert version
