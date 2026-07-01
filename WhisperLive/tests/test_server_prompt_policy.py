from __future__ import annotations

import sys
import types


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self


def _install_server_import_stubs() -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _Dummy
    fastapi.UploadFile = _Dummy
    fastapi.Form = lambda *args, **kwargs: None
    fastapi.Request = _Dummy
    fastapi.File = lambda *args, **kwargs: None
    sys.modules.setdefault("fastapi", fastapi)

    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = _Dummy
    sys.modules.setdefault("fastapi.middleware", types.ModuleType("fastapi.middleware"))
    sys.modules.setdefault("fastapi.middleware.cors", cors)

    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _Dummy
    sys.modules.setdefault("fastapi.responses", responses)

    starlette_responses = types.ModuleType("starlette.responses")
    starlette_responses.PlainTextResponse = _Dummy
    starlette_responses.StreamingResponse = _Dummy
    sys.modules.setdefault("starlette", types.ModuleType("starlette"))
    sys.modules.setdefault("starlette.responses", starlette_responses)

    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    sys.modules.setdefault("uvicorn", uvicorn)

    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = _Dummy
    sys.modules.setdefault("faster_whisper", faster_whisper)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)

    websockets_server = types.ModuleType("websockets.sync.server")
    websockets_server.serve = lambda *args, **kwargs: _Dummy()
    sys.modules.setdefault("websockets", types.ModuleType("websockets"))
    sys.modules.setdefault("websockets.sync", types.ModuleType("websockets.sync"))
    sys.modules.setdefault("websockets.sync.server", websockets_server)

    websockets_exceptions = types.ModuleType("websockets.exceptions")
    websockets_exceptions.ConnectionClosed = ConnectionError
    sys.modules.setdefault("websockets.exceptions", websockets_exceptions)

    onnxruntime = types.ModuleType("onnxruntime")
    onnxruntime.InferenceSession = _Dummy
    sys.modules.setdefault("onnxruntime", onnxruntime)


_install_server_import_stubs()

from whisper_live.server import TranscriptionServer


def test_server_prompt_policy_forces_server_defaults() -> None:
    server = TranscriptionServer()
    server.force_server_prompt = True
    server.default_initial_prompt = "server prompt"

    value = server._option_with_server_default(
        {"initial_prompt": "client prompt"},
        "initial_prompt",
        server.default_initial_prompt,
    )

    assert value == "server prompt"


def test_server_prompt_policy_allows_client_override_when_not_forced() -> None:
    server = TranscriptionServer()
    server.force_server_prompt = False

    value = server._option_with_server_default(
        {"initial_prompt": "client prompt"},
        "initial_prompt",
        "server prompt",
    )

    assert value == "client prompt"


def test_server_prompt_policy_uses_default_when_client_omits_value() -> None:
    server = TranscriptionServer()
    server.force_server_prompt = False

    value = server._option_with_server_default({}, "initial_prompt", "server prompt")

    assert value == "server prompt"
