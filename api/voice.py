import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from _lib.engine import process_voice_chat

app = Flask(__name__)
application = app


@app.route("/api/voice", methods=["POST"])
def voice():
    try:
        user_id = request.form.get("user_id") or request.form.get("session_id")
        if not user_id:
            # allow JSON callers too, in case user_id is sent as a query/json field
            user_id = request.args.get("user_id")

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        audio_file = request.files.get("audio") or request.files.get("audio_file")
        if not audio_file:
            return jsonify({"error": "audio file is required (form field 'audio')"}), 400

        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"error": "uploaded audio file is empty"}), 400

        filename = audio_file.filename or "voice_note"
        mime_type = audio_file.mimetype or "audio/wav"

        # rudo-chat's callRudoVoiceBackend sends this as the full language
        # name ("english"/"shona"/...), matching what process_voice_chat
        # checks against VOICE_SUPPORTED_LANGUAGES — not the 2-letter
        # "en"/"sn" code used on the Flutter/edge-function side. Falls back
        # to None (not a hardcoded default) so process_voice_chat's own
        # first-message/Redis-state logic still applies for any caller that
        # doesn't send a language at all.
        language = request.form.get("language") or request.args.get("language")

        result = process_voice_chat(
            user_id, audio_bytes, filename=filename, mime_type=mime_type, language=language
        )
        return jsonify(result)
    except Exception as e:
        logging.error(f"/api/voice error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
