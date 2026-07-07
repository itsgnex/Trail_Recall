from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_models: dict[str, Any] = {}

def _get_model(name: str):
    model = _models.get(name)
    if model is not None:
        return model

    from faster_whisper import WhisperModel

    print(f"whisper worker loading model: {name}", file=sys.stderr)
    model = WhisperModel(name, device="auto", compute_type="int8")
    _models[name] = model
    return model


def _transcribe(wav_path: str, model_name: str, vad_filter: bool = False, initial_prompt: str | None = None) -> str:
    model = _get_model(model_name)
    segments, _ = model.transcribe(
        wav_path,
        language="en",
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> int:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            wav_path = str(request["wav_path"])
            if not Path(wav_path).exists():
                raise FileNotFoundError(wav_path)
            text = _transcribe(
                wav_path,
                str(request.get("model", "small.en")),
                bool(request.get("vad_filter", False)),
                request.get("initial_prompt"),
            )
            response = {"text": text}
        except Exception as exc:
            response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
