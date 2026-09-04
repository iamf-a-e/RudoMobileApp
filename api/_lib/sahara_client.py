"""
Dawa Health — Sahara STT client + voice-language gating
==========================================================
Sahara's TTS voice list currently only covers Shona (sn) and English (en)
out of engine.py's seven supported languages. Ndebele, Chinyanja, Bemba,
Tonga, and Lozi speakers who try voice get an English apology + redirect
to text, since Sahara has no way to synthesize speech in their language.

We deliberately do NOT use Sahara's get_answer=TRUE post-processing option
for transcription — that returns ungrounded generic-LLM answers, not
answers grounded in our own pregnancy_data / cervical_cancer_data content.
We transcribe only (get_answer=FALSE), then hand the transcript to the
existing engine.py pipeline (ask_gemini + our grounding helpers).
"""

import os
import time
import logging
import requests

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

SAHARA_API_KEY = os.environ.get("SAHARA_API_KEY")
SAHARA_UPLOAD_URL = "https://infer.voice.intron.io/file/v1/upload/sync"
SAHARA_STATUS_URL = "https://infer.voice.intron.io/file/v1/status"  # confirm exact path in "Get File Status" docs

if not SAHARA_API_KEY:
    logging.warning("SAHARA_API_KEY environment variable not set — Sahara calls will fail")

# Confirmed against Intron's "Supported Languages And Accents" page
# (Afrikaans, Amharic, English, Hausa, Igbo, Kinyarwanda, Luganda, Oromo,
# Pidgin, Shona, Swahili, Wolof, Yoruba). Of engine.py's seven languages,
# only english and shona are on that list.
VOICE_SUPPORTED_LANGUAGES = {"english", "shona"}

ENGINE_TO_SAHARA_LANG = {
    "english": "en",
    "shona": "sn",
    # ndebele, chinyanja, bemba, tonga, lozi intentionally omitted —
    # not in Sahara's supported language list as of this writing
}

VOICE_UNSUPPORTED_MESSAGE = (
    "Sorry, voice isn't available in your language yet. Please type your message instead, "
    "or continue by voice in English or Shona."
)


# ─────────────────────────────────────────────
#  Voice fallback for languages Sahara doesn't support
# ─────────────────────────────────────────────

def get_voice_unsupported_response(language):
    """
    Called instead of hitting Sahara at all when we already know the
    user's language isn't voice-supported. Always English — Sahara has
    no TTS voice to speak this back in the user's own language.
    """
    return {
        "reply": VOICE_UNSUPPORTED_MESSAGE,
        "reply_audio": None,
        "language": language,
        "voice_supported": False,
    }


# ─────────────────────────────────────────────
#  Transcription
# ─────────────────────────────────────────────

def transcribe_audio(audio_bytes, filename="voice_note", mime_type="audio/wav", language_hint=None):
    """
    Transcribe audio via Sahara. Returns (transcript_text, file_id) or (None, None).
    Handles the documented 503-timeout-with-file_id case by polling status.
    Caller is responsible for only calling this when language_hint (if known)
    is in VOICE_SUPPORTED_LANGUAGES — see process_voice_chat in engine.py.
    """
    if not SAHARA_API_KEY:
        logging.error("SAHARA_API_KEY not set")
        return None, None

    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}"}
    files = {"audio_file_blob": (filename, audio_bytes, mime_type)}
    data = {
        "audio_file_name": filename,
        "use_category": "file_category_general",
        "get_answer": "FALSE",
    }
    sahara_lang = ENGINE_TO_SAHARA_LANG.get(language_hint)
    if sahara_lang:
        data["use_language_asr_input"] = sahara_lang

    try:
        resp = requests.post(SAHARA_UPLOAD_URL, headers=headers, files=files, data=data, timeout=15)

        if resp.status_code == 503:
            # Documented behavior: still processing after 120s — file_id
            # comes back in the body, poll Get File Status with it.
            payload = resp.json()
            file_id = payload.get("data", {}).get("file_id") or payload.get("file_id")
            if file_id:
                return _poll_transcript(file_id, headers)
            logging.error("[transcribe_audio] 503 with no file_id in response")
            return None, None

        resp.raise_for_status()
        payload = resp.json()
        transcript = payload.get("data", {}).get("audio_transcript", "")
        file_id = payload.get("data", {}).get("file_id")
        return transcript, file_id

    except Exception as e:
        logging.error(f"[transcribe_audio] {type(e).__name__}: {e}")
        return None, None


def _poll_transcript(file_id, headers, max_attempts=10, delay_seconds=3):
    """Poll Get File Status until FILE_TRANSCRIBED or we give up."""
    for _ in range(max_attempts):
        time.sleep(delay_seconds)
        try:
            resp = requests.get(f"{SAHARA_STATUS_URL}/{file_id}", headers=headers, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", {})
            if data.get("processing_status") == "FILE_TRANSCRIBED":
                return data.get("audio_transcript", ""), file_id
        except Exception as e:
            logging.error(f"[_poll_transcript] {type(e).__name__}: {e}")
    logging.error(f"[_poll_transcript] gave up waiting on file_id={file_id}")
    return None, file_id
