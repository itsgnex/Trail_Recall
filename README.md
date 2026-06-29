# Gaze-Triggered Plant and Sign Assistant

Phase 1 Mac-only webcam prototype.

## Run

```bash
python --version
python -m pip install -r requirements.txt
python -m py_compile main.py
python main.py
```

Press `q` in the camera window to quit.

Run without a camera flag to scan indexes 0 through 10, preview available cameras, and choose one:

```bash
python main.py
```

Or open a camera directly:

```bash
python main.py --camera 0
python main.py --camera 1
python main.py --camera 2
python main.py --camera 2 --mic 0
```

## Optional AI

The app works without CLIP or Ollama.

To try local Ollama models:

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
```

Config switches:

```bash
USE_CLIP=true
USE_LLM_INTENT=true
USE_OLLAMA=true
DIALOGUE_MODEL=gemma3:1b
VISION_LLM_MODEL=gemma3:4b
FINAL_RESPONSE_MODEL=gemma3:4b
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_DIALOGUE_TIMEOUT=30
OLLAMA_TEXT_TIMEOUT=60
OLLAMA_IMAGE_TIMEOUT=180
USE_WHISPER_STT=true
WHISPER_MODEL=small.en
WHISPER_RECORD_SECONDS=7
PLANT_ID_PROVIDER=plantnet
PLANTNET_API_KEY=
USE_PLANT_ID=false
MIC_DEVICE_INDEX=0
MIC_LISTEN_TIMEOUT=12
MIC_PHRASE_TIME_LIMIT=7
MIC_AMBIENT_NOISE_DURATION=1
```

If Ollama is missing or not running, the app prints one warning and uses local fallback intent and response templates.

Gemma 3 1B handles fast dialogue routing. Gemma 3 4B is used only after the dwell trigger and user reply for image understanding and final grounded responses. The app does not send every webcam frame to Ollama.

When a trigger fires, the app saves the current crop in `debug_crops/` and sends that crop to Gemma 3 4B for sign or plant analysis after the user confirms.

Plant ID is optional. If `PLANTNET_API_KEY` is empty, plant species identification is skipped and Gemma/fallback responses clearly say the exact type is uncertain.

CLIP/OpenCLIP is optional. If `open_clip` or PyTorch is missing or not compatible with your Python version, the app prints one warning and falls back to OCR plus simple image heuristics. On Python 3.13, PyTorch/OpenCLIP wheels may be limited; the prototype still starts without them.

Optional tools:

- Install Tesseract for better sign reading: `brew install tesseract`
- Install PyAudio for microphone input if `SpeechRecognition` reports it is missing: `brew install portaudio && python -m pip install pyaudio`
- If microphone access fails, allow microphone permission in macOS. Typed input fallback still works.
