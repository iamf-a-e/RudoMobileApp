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

CODE-SWITCHING NOTE (added while diagnosing "Rudo doesn't understand
Shona/English code-switching" bug):
  transcribe_audio always forces the ENTIRE audio clip to be decoded as a
  single language via `use_language_asr_input` — whatever language_hint
  (itself just the last language detect_language() settled on for this
  user, in engine.py's Redis-backed state) resolves to. There is currently
  no code-switching / auto-detect mode wired up here. For a user who mixes
  English and Shona mid-utterance ("ndiri kuda ku-book appointment for
  2pm"), forcing a single-language hint mangles whichever portion doesn't
  match the hint — this looks exactly like generic STT noise/garbling but
  is actually a language-hint mismatch, not a model-quality problem.

  Intron (Sahara's vendor) released Sahara v2.5 in beta around August
  2026 with genuine bilingual code-switching support across roughly a
  dozen African-language pairs (confirmed: Zulu, Hausa, Swahili, Luganda,
  Igbo, plus a Kinyarwanda/English/French trilingual model) — but as of
  this writing it is UNCONFIRMED whether (a) that capability is available
  on this specific endpoint (SAHARA_UPLOAD_URL, the general-purpose sync
  upload endpoint) rather than only via a separate challenge/beta API
  surface, and (b) whether Shona specifically is among the supported
  code-switch pairs. Before changing use_language_asr_input logic here
  (e.g. omitting it for auto-detect, or switching to a code-switch-aware
  parameter/model), confirm both of those directly with Intron — this is
  not something to guess at from public marketing material alone.

  The logging added to transcribe_audio below records the resolved hint
  actually sent per request, so real stuck transcripts can be checked
  against it to confirm (or rule out) the forced-hint theory before
  changing behavior.
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

    # DIAGNOSTIC: as of this writing, this always forces the ENTIRE audio
    # clip to be decoded as a single language (whatever language_hint
    # resolves to) — there is currently no code-switching / auto-detect
    # mode wired up here. That's a real problem for users who mix English
    # and Shona mid-utterance: a forced hint of "english" will mangle the
    # Shona portions (and vice versa), which looks like generic STT noise
    # but is actually a language-hint mismatch. This log line makes that
    # visible per-request so it can be confirmed against real transcripts
    # rather than assumed. See the CODE-SWITCHING NOTE at the top of this
    # file before changing use_language_asr_input logic — check with
    # Intron whether Sahara v2.5's code-switching support is available on
    # this endpoint and covers Shona before assuming omitting the hint (or
    # any other change) is the right fix.
    logging.info(
        f"[transcribe_audio] language_hint={language_hint} sahara_lang={sahara_lang} "
        f"audio_bytes={len(audio_bytes)} mime_type={mime_type} filename={filename}"
    )

    try:
        resp = requests.post(SAHARA_UPLOAD_URL, headers=headers, files=files, data=data, timeout=15)

        if not resp.ok:
            logging.error(
                f"[transcribe_audio] HTTP {resp.status_code} "
                f"headers={dict(resp.headers)} body={resp.text[:2000]}"
            )
            
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
