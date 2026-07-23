#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_http import synthesize_wav  # noqa: E402
from src.common_tts import AUDIO_DIR, MANIFEST_PATH, base_manifest, load_manifest, save_manifest, validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate persistent common TrailRecall TTS WAV files.")
    parser.add_argument("--force", action="store_true", help="Regenerate every static common WAV.")
    args = parser.parse_args()

    old_manifest = load_manifest()
    manifest = base_manifest()
    manifest["generatedTemplates"] = [
        {**entry, "wavFilename": entry.get("wavFilename") or entry.get("wav", "")}
        for entry in old_manifest.get("generatedTemplates", [])
    ]
    save_manifest(manifest)
    validation = validate_manifest(manifest)
    if not validation["ok"]:
        for error in validation["errors"]:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    generated = 0
    skipped = 0
    for entry in manifest["phrases"]:
        wav_path = AUDIO_DIR / entry["wav"]
        if wav_path.exists() and not args.force:
            skipped += 1
            continue
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(synthesize_wav(entry["text"], sample_rate=manifest["sampleRate"], channels=manifest["channels"]))
        generated += 1
        print(f"generated {entry['phraseId']} -> {wav_path}")

    print(f"COMMON_TTS_GENERATION\nmanifest={MANIFEST_PATH}\naudioDir={AUDIO_DIR}\ngenerated={generated}\nskipped={skipped}\nphrases={len(manifest['phrases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
