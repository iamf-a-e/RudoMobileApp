import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from _lib.engine import get_history

app = Flask(__name__)
application = app


@app.route("/api/chat-history", methods=["GET"])
def chat_history():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id query parameter is required"}), 400
        limit = request.args.get("limit", default=50, type=int)
        history = get_history(user_id, limit)
        return jsonify({"user_id": user_id, "messages": history, "count": len(history)})
    except Exception as e:
        logging.error(f"/api/chat-history error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
