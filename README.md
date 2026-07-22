# Gaze-Triggered Plant and Sign Assistant

Phase 1 Mac-only webcam prototype.

## Run

Use the checked-in virtualenv if available:

```bash
./.venv312/bin/python --version
./.venv312/bin/python -m pip install -r requirements.txt
./.venv312/bin/python -m py_compile main.py
```

### Camera-only demo mode

This is the simplest way to open the camera and see a clear ready signal:

```bash
./.venv312/bin/python main.py --camera 2 --camera-only
```

You should see:
- a camera window
- a big `READY - camera-only demo` banner on the video
- `READY: camera-only preview is live. Press q to quit.` in the terminal

Press `q` in the camera window to quit.

### Full mode with mic

If you want the normal Mentra glasses flow over the phone hotspot:

```bash
./scripts/start_mentra_stack.sh
```

Or run the Python app directly in Mentra stream mode:

```bash
export ANDROID_TRAIL_URL="http://192.168.0.91:8766"
./.venv312/bin/python main.py --camera-source mentra
```

Open a local Mac camera only when explicitly requested:

```bash
./.venv312/bin/python main.py --camera-source local --camera-index 2
```

Or point the app at a stream URL directly:

```bash
./.venv312/bin/python main.py --camera-source rtmp://192.168.1.65:1935/live/mentra-live --mic 0
./.venv312/bin/python main.py --camera-source http://192.168.1.65:8888/live/mentra-live/index.m3u8 --mic 0
```

Use `--camera-source` when you want the Mentra stream to be used as the camera input without OBS.

Diagnostics:

```bash
./.venv312/bin/python main.py --network-check
./.venv312/bin/python main.py --camera-source mentra --preview-only
./scripts/run_rtmp_server.sh probe
```

## Optional AI

The app works without CLIP or Ollama.

To try local Ollama models:

```bash
ollama pull gemma3:1b
```

Config switches:

```bash
VOICE_PROFILE=balanced
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_TIMEOUT=4
FINAL_RESPONSE_MODEL=gemma3:1b
IMAGE_TASK_MODEL=gemma3:1b
USE_LOCAL_VISION_LLM=false
STT_PROVIDER=local
OPENAI_API_KEY=
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_STT_TIMEOUT=8
USE_CLIP=true
USE_LLM_INTENT=true
USE_OLLAMA=true
DIALOGUE_MODEL=gemma3:1b
VISION_LLM_MODEL=gemma3:1b
FINAL_RESPONSE_MODEL=gemma3:1b
OLLAMA_MODEL=gemma3:1b
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_DIALOGUE_TIMEOUT=30
OLLAMA_TEXT_TIMEOUT=60
OLLAMA_IMAGE_TIMEOUT=180
USE_WHISPER_STT=true
WHISPER_MODEL=small.en
WHISPER_RECORD_SECONDS=7
WHISPER_SHORT_REPLY_MODE=true
WHISPER_USE_VAD=false
WHISPER_INITIAL_PROMPT=yes, no, sure, why not, please do, don't, not now, can you do it, tell me more, repeat that
FOLLOW_UP_MODE=true
FOLLOW_UP_TIMEOUT_SECONDS=8
MAX_FOLLOW_UP_TURNS=1
SPEAK_FOLLOW_UP_OFFER=false
FOLLOW_UP_SILENCE_RETURNS_TO_SCAN=true
WAKE_MODE=false
PLANT_ID_PROVIDER=plantnet
PLANTNET_API_KEY=
USE_PLANT_ID=false
MIC_DEVICE_INDEX=0
MIC_LISTEN_TIMEOUT=12
MIC_PHRASE_TIME_LIMIT=7
MIC_AMBIENT_NOISE_DURATION=1
ANDROID_TRAIL_URL=http://192.168.0.91:8766
```

The camera crop sent to vision is currently resized to a **384px max edge** JPEG before being encoded and sent.

Voice profiles:

- `VOICE_PROFILE=fast`: smallest Whisper and shortest wake/command windows
- `VOICE_PROFILE=whisper_first`: Whisper first, Google only as backup
- `VOICE_PROFILE=balanced`: current default behavior

Speech-to-text providers:

- `STT_PROVIDER=local`: use local faster-whisper
- `STT_PROVIDER=openai`: skip local Whisper and use OpenAI speech-to-text
- `STT_PROVIDER=openai_first`: try OpenAI first, then local Whisper if the API fails

If Ollama is missing or not running, the app prints one warning and uses local fallback intent and response templates.

Gemma 3 1B handles fast dialogue routing and short text responses. OpenRouter handles image understanding after the dwell trigger and user reply. The app does not send every webcam frame to an API.

When a trigger fires, the app saves the current crop in `debug_crops/` and sends that crop to OpenRouter for sign or plant analysis after the user confirms.

## PlantNet

Plant species identification is optional. Set your key before running:

```bash
cp .env.example .env
```

Then edit `.env` and add your real key:

```bash
PLANTNET_API_KEY=your_real_key_here
USE_PLANT_ID=true
PLANT_ID_PROVIDER=plantnet
PLANTNET_PROJECT=all
PLANTNET_LANG=en
PLANTNET_ORGAN=auto
PLANTNET_MIN_CONFIDENCE=0.45
PLANTNET_HIGH_CONFIDENCE=0.70
```

Direct PlantNet test:

```bash
python -m src.plant_id debug_crops/plant_example.jpg
```

Confidence guide:

- `>= 0.70`: high confidence
- `0.45` to `< 0.70`: possible, but not certain
- `< 0.45`: do not claim the exact species

If `PLANTNET_API_KEY` is empty, plant species identification is skipped and Gemma/fallback responses clearly say the exact type is uncertain.

Follow-up listening uses a configurable timeout because older-adult voice interaction research suggests response timing should be adjustable. The prototype defaults to 8 seconds and can be tested at 6, 8, or 10 seconds during evaluation.

CLIP/OpenCLIP is optional. If `open_clip` or PyTorch is missing or not compatible with your Python version, the app prints one warning and falls back to OCR plus simple image heuristics. On Python 3.13, PyTorch/OpenCLIP wheels may be limited; the prototype still starts without them.

Optional tools:

- Install Tesseract for better sign reading: `brew install tesseract`
- Install PyAudio for microphone input if `SpeechRecognition` reports it is missing: `brew install portaudio && python -m pip install pyaudio`
- If microphone access fails, allow microphone permission in macOS. Typed input fallback still works.
