import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from _lib.engine import process_chat

app = Flask(__name__)
application = app


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id") or data.get("session_id")
        message = data.get("message")
        forced_lang = data.get("language")

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        result = process_chat(user_id, message, forced_lang)
        return jsonify({
            "success": True,
            "message": "Message sent successfully",
            **result,
        })
    except Exception as e:
        logging.error(f"/api/send-message error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
