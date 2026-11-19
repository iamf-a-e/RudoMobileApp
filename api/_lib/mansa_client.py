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

Audio handling (browser recordings):
  1. The app can mislabel what it recorded (for example WAV bytes sent as
     audio/mp4). Mansa trusts the label, decodes the wrong codec and returns
     garbage such as Arabic script for Shona speech. transcribe_audio detects
     the real format from the file header and ignores a wrong label.
  2. Empty recordings (header only, about 44 bytes) are rejected before calling
     Mansa, which would otherwise fail with "Could not determine audio duration".
  3. prepare_audio() is public so engine.py can run language identification on
     the same cleaned audio before transcribing.
  4. WAV audio keeps its original sample rate and channels by default. An
     optional 16 kHz mono conversion exists (MANSA_TARGET_SAMPLE_RATE) but made
     transcripts worse in testing.
  5. Near-silent clips are rejected (see SILENCE_PEAK) because speech models
     hallucinate text for them. Loudness is logged on every request.
  6. Quiet recordings are amplified toward a healthy level (see NORMALIZE_PEAK).
  7. Browser WAV headers can carry bad chunk sizes (0 or 0xFFFFFFFF) or
     32-bit float samples. Mansa then fails with "Could not determine audio
     duration". _normalize_wav rebuilds a clean 16-bit PCM WAV and measures the
     real duration.
"""

import io
import os
import re
import base64
import logging
import struct
import sys
import wave
from array import array
from itertools import accumulate

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

# Optional conversion of WAV input to 16 kHz mono (set MANSA_TARGET_SAMPLE_RATE=16000
# to turn it on). It is OFF by default: in a side by side test on the same
# recording, Mansa transcribed the original 44.1 kHz stereo audio better than a
# 16 kHz mono conversion. 0 means send the original sample rate and channels.
TARGET_SAMPLE_RATE = int(os.environ.get("MANSA_TARGET_SAMPLE_RATE", "0"))

# A clip whose loudest sample is below this fraction of full scale is silence or
# mic noise. Speech models invent text for such clips (a real example was a
# Welsh sentence about Bhutan), so they are rejected instead of transcribed.
SILENCE_PEAK = float(os.environ.get("MANSA_SILENCE_PEAK", "0.01"))

# Quiet recordings (laptop mics, distance from the phone) transcribe worse.
# WAV audio whose loudest sample is below NORMALIZE_PEAK is amplified so that
# peak reaches NORMALIZE_PEAK, never by more than MAX_GAIN times. Clips quieter
# than SILENCE_PEAK are never boosted (they are rejected as silence). Set
# MANSA_NORMALIZE_PEAK=0 to turn this off.
NORMALIZE_PEAK = float(os.environ.get("MANSA_NORMALIZE_PEAK", "0.7"))
MAX_GAIN = float(os.environ.get("MANSA_MAX_GAIN", "20"))

# Anything smaller than this is a header with no speech (a bare WAV header is 44
# bytes). Mansa answers those with "Could not determine audio duration".
MIN_AUDIO_BYTES = 2000

# All four STT languages are written in Latin script. A transcript in Arabic
# script for one of them means the audio was decoded wrongly, not that the
# patient spoke Arabic.
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


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


def _sniff_mime(audio_bytes, claimed):
    """Detect the real audio format from the file header, ignoring the claimed type."""
    head = audio_bytes[:16]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"
    if head[:4] == b"fLaC":
        return "audio/flac"
    if head[4:8] == b"ftyp":
        return "audio/mp4"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    return claimed


def _to_mono_resampled(data, ch, rate, target_rate):
    """
    Mix 16-bit PCM down to mono and, if the rate is above target_rate, lower it
    with a box filter (an average over each output sample's window, which acts
    as a crude anti-alias filter). Pure Python so it runs on Vercel without
    numpy or audioop. Returns (pcm_bytes, 1, new_rate), or None on bad input.
    """
    samples = array("h")
    samples.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()

    if ch > 1:
        usable = len(samples) // ch * ch
        channels = [samples[c:usable:ch] for c in range(ch)]
        mono = array("h", (sum(frame) // ch for frame in zip(*channels)))
    else:
        mono = samples

    new_rate = rate
    if target_rate and rate > target_rate and len(mono) > 1:
        ratio = rate / float(target_rate)
        half = ratio / 2.0
        total = len(mono)
        cs = [0]
        cs.extend(accumulate(mono))
        out = array("h")
        for j in range(int(total / ratio)):
            centre = (j + 0.5) * ratio
            a = max(0, int(centre - half))
            b = min(total, max(a + 1, int(centre + half)))
            out.append(int((cs[b] - cs[a]) / (b - a)))
        mono = out
        new_rate = target_rate

    if not mono:
        return None
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes(), 1, new_rate


def _boost_quiet_speech(data):
    """Amplify quiet 16-bit PCM toward NORMALIZE_PEAK. Returns PCM bytes."""
    if not NORMALIZE_PEAK or not data:
        return data
    samples = array("h")
    samples.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return data
    peak = max(max(samples), -min(samples)) / 32768.0
    if peak < SILENCE_PEAK or peak >= NORMALIZE_PEAK * 0.9:
        return data
    gain = min(MAX_GAIN, NORMALIZE_PEAK / peak)
    if gain < 1.2:
        return data
    boosted = array(
        "h", (max(-32768, min(32767, int(x * gain))) for x in samples)
    )
    logging.info(
        f"[prepare_audio] quiet clip (peak={peak:.3f}), applied gain x{gain:.1f}"
    )
    if sys.byteorder == "big":
        boosted.byteswap()
    return boosted.tobytes()


def _normalize_wav(audio_bytes):
    """
    Rebuild a clean 16-bit PCM WAV from browser-recorded audio. Fixes bad chunk
    sizes (0 or 0xFFFFFFFF) and converts 32-bit float samples to 16-bit.
    Returns (wav_bytes, duration_seconds), or None if it can't be parsed.
    """
    b = audio_bytes
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return None

    fmt = None
    data = None
    off = 12
    while off + 8 <= len(b):
        cid = b[off:off + 4]
        size = struct.unpack_from("<I", b, off + 4)[0]
        body = off + 8
        if cid == b"fmt " and body + 16 <= len(b):
            code, ch, rate, _, _, bits = struct.unpack_from("<HHIIHH", b, body)
            if code == 0xFFFE and body + 26 <= len(b):  # WAVE_FORMAT_EXTENSIBLE
                code = struct.unpack_from("<H", b, body + 24)[0]
            fmt = (code, ch, rate, bits)
        elif cid == b"data":
            remaining = len(b) - body
            if size == 0 or size > remaining:
                size = remaining
            data = b[body:body + size]
            break
        off = body + size + (size % 2)

    if not fmt or data is None:
        return None
    code, ch, rate, bits = fmt

    if code == 3 and bits == 32:  # float32 -> int16
        samples = array("f")
        samples.frombytes(data[: len(data) // 4 * 4])
        data = array(
            "h", (max(-32768, min(32767, int(s * 32767))) for s in samples)
        ).tobytes()
        code, bits = 1, 16
    if code != 1 or ch < 1 or rate < 1 or bits % 8:
        return None

    block = ch * bits // 8
    data = data[: len(data) // block * block]
    if not data:
        return None

    # Speech format: 16 kHz mono. Only for 16-bit PCM, which is what browsers
    # produce and what float32 input was converted to above.
    if bits == 16 and TARGET_SAMPLE_RATE and (ch > 1 or rate > TARGET_SAMPLE_RATE):
        converted = _to_mono_resampled(data, ch, rate, TARGET_SAMPLE_RATE)
        if converted:
            data, ch, rate = converted
            block = ch * bits // 8

    if bits == 16:
        data = _boost_quiet_speech(data)

    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, rate, rate * block, block, bits)
        + b"data" + struct.pack("<I", len(data))
    )
    return header + data, len(data) / float(rate * block)


def _clean_for_tts(text):
    """Strip markdown noise, collapse whitespace, cap length at a sentence boundary."""
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= TTS_MAX_CHARS:
        return text
    cut = text[:TTS_MAX_CHARS]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > TTS_MAX_CHARS // 2 else cut


def prepare_audio(audio_bytes, claimed_mime, duration_seconds=None):
    """
    Make browser or phone audio safe to hand to any speech model.
    Detects the real format (ignoring a wrong label) and, for WAV, rebuilds a
    clean 16-bit PCM file with the measured duration.
    Returns (audio_bytes, mime_type, duration_seconds). Safe to call twice.
    """
    claimed = (claimed_mime or "audio/wav").split(";")[0].strip()
    mime_type = _sniff_mime(audio_bytes, claimed)
    if mime_type != claimed:
        logging.warning(
            f"[prepare_audio] mime mismatch claimed={claimed} detected={mime_type}"
        )

    if mime_type == "audio/wav":
        normalized = _normalize_wav(audio_bytes)
        if normalized:
            audio_bytes, duration_seconds = normalized
            logging.info(
                f"[prepare_audio] normalized wav: {len(audio_bytes)} bytes, "
                f"{duration_seconds:.1f}s"
            )
        else:
            logging.warning("[prepare_audio] could not normalize wav, sending as is")

    return audio_bytes, mime_type, duration_seconds


def measure_levels(audio_bytes, mime_type="audio/wav"):
    """
    Loudness of a 16-bit PCM WAV: {"peak", "rms", "seconds"} as fractions of
    full scale (peak and rms) and seconds of audio. Returns None for other
    formats or if it can't be read. Speech normally peaks above 0.05.
    """
    if mime_type != "audio/wav":
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            if w.getsampwidth() != 2:
                return None
            frames = w.readframes(w.getnframes())
            rate = w.getframerate() * w.getnchannels()
        samples = array("h")
        samples.frombytes(frames[: len(frames) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        if not samples:
            return None
        peak = max(max(samples), -min(samples)) / 32768.0
        rms = (sum(x * x for x in samples) / len(samples)) ** 0.5 / 32768.0
        return {"peak": peak, "rms": rms, "seconds": len(samples) / float(rate)}
    except Exception as e:
        logging.warning(f"[measure_levels] {type(e).__name__}: {e}")
        return None


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

    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        logging.warning(
            f"[transcribe_audio] audio too short to contain speech "
            f"(audio_bytes={len(audio_bytes or b'')}), not calling Mansa"
        )
        return None

    # Trust the bytes over the label the app sent, and rebuild a clean WAV.
    audio_bytes, mime_type, duration_seconds = prepare_audio(
        audio_bytes, mime_type, duration_seconds
    )

    levels = measure_levels(audio_bytes, mime_type)
    if levels:
        logging.info(
            f"[transcribe_audio] levels peak={levels['peak']:.3f} rms={levels['rms']:.4f} "
            f"seconds={levels['seconds']:.1f}"
        )
        if levels["peak"] < SILENCE_PEAK:
            logging.warning(
                f"[transcribe_audio] clip is near silent (peak={levels['peak']:.4f}), "
                "not calling Mansa"
            )
            return None

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
        text = ((resp.json().get("data") or {}).get("transcript") or "").strip()
    except Exception as e:
        logging.error(f"[transcribe_audio] {type(e).__name__}: {e}")
        return None

    # Arabic script for a Latin-script language means a bad decode. Return None
    # so the caller can ask the patient to try again instead of sending garbage
    # to the medical model.
    if text and _ARABIC_SCRIPT.search(text):
        logging.warning(
            f"[transcribe_audio] Arabic-script transcript for language_hint={language_hint}, "
            f"rejecting (mime_type={mime_type}, audio_bytes={len(audio_bytes)})"
        )
        return None

    return text


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
