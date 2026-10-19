"""
Dawa Health: Mansa STT/TTS client + voice-language gating
==========================================================
Replaces sahara_client.py. Lives at api/_lib/mansa_client.py.

Endpoints (base URL comes from MANSA_API_BASE_URL, copy it from the Mansa console):
    POST /v1/transcribe     JSON in, JSON out (data.transcript)
    POST /v1/audio/speech   JSON in, binary audio/wav out

Coverage versus engine.py's seven languages:
    STT: english, shona, bemba, chinyanja (Mansa calls it Chichewa)
    TTS: the four above plus ndebele
    Neither: tonga, lozi

Voice input needs STT, so VOICE_SUPPORTED_LANGUAGES is the STT set. Ndebele has
TTS only, so it is kept in TTS_LANGUAGES for later but gets the text fallback today.

Mansa requires an explicit language for every transcription. There is no
documented auto-detect or code-switching mode, so a clip is decoded as ONE
language (the same single-language limitation the Sahara client had).
"""

import io
import os
import re
import base64
import logging
import wave

import requests

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

MANSA_API_KEY = os.environ.get("MANSA_API_KEY")
MANSA_API_BASE_URL = (os.environ.get("MANSA_API_BASE_URL") or "").rstrip("/")

# Mansa bills TTS per input character, so cap what we synthesize per reply.
TTS_MAX_CHARS = int(os.environ.get("MANSA_TTS_MAX_CHARS", "800"))

if not MANSA_API_KEY or not MANSA_API_BASE_URL:
    logging.warning("MANSA_API_KEY / MANSA_API_BASE_URL not set. Mansa calls will fail")

# engine.py language key -> Mansa STT value (case-sensitive)
STT_LANGUAGES = {
    "english": "English",
    "shona": "Shona",
    "bemba": "Bemba",
    "chinyanja": "Chichewa",
}

# engine.py language key -> Mansa TTS slug
TTS_LANGUAGES = {
    "english": "english",
    "shona": "shona",
    "bemba": "bemba",
    "chinyanja": "chichewa",
    "ndebele": "ndebele",
}

# English gets the South African female accent. Every other language uses the
# plain "female" voice, which speaks that language natively. Do NOT use the
# "{language}_female_speaker" voices: those speak English with that accent.
TTS_VOICES = {"english": "south_african_female"}
DEFAULT_VOICE = "female"

VOICE_SUPPORTED_LANGUAGES = set(STT_LANGUAGES)

# Codes and alternate names callers might send
LANGUAGE_ALIASES = {
    "en": "english",
    "sn": "shona",
    "bem": "bemba",
    "ny": "chinyanja",
    "nya": "chinyanja",
    "nyanja": "chinyanja",
    "chichewa": "chinyanja",
}

VOICE_UNSUPPORTED_MESSAGE = (
    "Sorry, voice isn't available in your language yet. Please type your message instead, "
    "or use voice in English, Shona, Bemba or Nyanja"
)

_MIME_EXT = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac",
    "audio/ogg": "ogg", "audio/webm": "webm", "audio/flac": "flac",
}


def normalize_language(language):
    key = (language or "").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


# ─────────────────────────────────────────────
#  Voice fallback for unsupported languages
# ─────────────────────────────────────────────

def get_voice_unsupported_response(language):
    """Returned instead of calling Mansa when the language has no STT."""
    return {
        "reply": VOICE_UNSUPPORTED_MESSAGE,
        "reply_audio": None,
        "language": language,
        "voice_supported": False,
    }


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _headers():
    return {
        "Authorization": f"Bearer {MANSA_API_KEY}",
        "Content-Type": "application/json",
    }


def _audio_duration(audio_bytes, mime_type, duration_seconds=None):
    """Seconds of audio, used by Mansa for billing. Prefer the caller's value."""
    if duration_seconds and duration_seconds > 0:
        return float(duration_seconds)
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        pass
    # Compressed formats (m4a/ogg/webm): rough estimate at ~6 KB/s of speech audio.
    est = max(1.0, len(audio_bytes) / 6000.0)
    logging.warning(
        f"[transcribe_audio] duration unknown for {mime_type}, estimating {est:.1f}s. "
        "Send duration_seconds from the app for accurate billing"
    )
    return est


def _filename_for(filename, mime_type):
    """Mansa wants an extension that matches the audio format."""
    ext = _MIME_EXT.get((mime_type or "").split(";")[0].strip().lower(), "wav")
    base = os.path.splitext(filename or "voice_note")[0] or "voice_note"
    return f"{base}.{ext}"


def _clean_for_tts(text):
    """Strip markdown noise, collapse whitespace, cap length at a sentence boundary."""
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= TTS_MAX_CHARS:
        return text
    cut = text[:TTS_MAX_CHARS]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > TTS_MAX_CHARS // 2 else cut


# ─────────────────────────────────────────────
#  Transcription
# ─────────────────────────────────────────────

def transcribe_audio(audio_bytes, filename="voice_note", mime_type="audio/wav",
                     language_hint="english", duration_seconds=None):
    """
    Transcribe audio via Mansa. Returns the transcript string, or None on failure.
    language_hint is an engine.py language key and must be in VOICE_SUPPORTED_LANGUAGES.
    """
    if not MANSA_API_KEY or not MANSA_API_BASE_URL:
        logging.error("MANSA_API_KEY / MANSA_API_BASE_URL not set")
        return None

    mansa_lang = STT_LANGUAGES.get(language_hint)
    if not mansa_lang:
        logging.error(f"[transcribe_audio] no Mansa STT for language_hint={language_hint}")
        return None

    mime_type = (mime_type or "audio/wav").split(";")[0].strip()
    duration = _audio_duration(audio_bytes, mime_type, duration_seconds)
    payload = {
        "audioBase64": base64.b64encode(audio_bytes).decode("ascii"),
        "language": mansa_lang,
        "mimeType": mime_type,
        "filename": _filename_for(filename, mime_type),
        "durationSeconds": duration,
    }
    logging.info(
        f"[transcribe_audio] language_hint={language_hint} mansa_lang={mansa_lang} "
        f"audio_bytes={len(audio_bytes)} mime_type={mime_type} duration={duration:.1f}s"
    )

    try:
        resp = requests.post(f"{MANSA_API_BASE_URL}/v1/transcribe",
                             headers=_headers(), json=payload, timeout=60)
        if not resp.ok:
            logging.error(f"[transcribe_audio] HTTP {resp.status_code} body={resp.text[:2000]}")
        resp.raise_for_status()
        return ((resp.json().get("data") or {}).get("transcript") or "").strip()
    except Exception as e:
        logging.error(f"[transcribe_audio] {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────
#  Text to speech
# ─────────────────────────────────────────────

def synthesize_speech(text, language="english"):
    """Returns WAV bytes for text, or None on failure or unsupported language."""
    if not MANSA_API_KEY or not MANSA_API_BASE_URL:
        logging.error("MANSA_API_KEY / MANSA_API_BASE_URL not set")
        return None

    slug = TTS_LANGUAGES.get(language)
    text = _clean_for_tts(text)
    if not slug or not text:
        return None

    payload = {
        "text": text,
        "language": slug,
        "voice": TTS_VOICES.get(language, DEFAULT_VOICE),
        "stream": False,
    }
    logging.info(f"[synthesize_speech] language={language} voice={payload['voice']} chars={len(text)}")

    try:
        resp = requests.post(f"{MANSA_API_BASE_URL}/v1/audio/speech",
                             headers=_headers(), json=payload, timeout=60)
        if not resp.ok:
            logging.error(f"[synthesize_speech] HTTP {resp.status_code} body={resp.text[:2000]}")
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logging.error(f"[synthesize_speech] {type(e).__name__}: {e}")
        return None
