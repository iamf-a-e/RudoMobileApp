import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify

app = Flask(__name__)
application = app


@app.route("/api/index", methods=["GET"])
@app.route("/api", methods=["GET"])
def home():
    return jsonify({
        "service": "Dawa Health Mobile Chat API",
        "version": "2.1.0",
        "endpoints": {
            "POST /api/chat": "Main chat endpoint — body: {user_id, message, language?}",
            "POST /api/send-message": "Alias of /api/chat",
            "GET /api/chat-history?user_id=...&limit=50": "Get chat history",
            "POST /api/reset": "Reset a user's conversation state — body: {user_id}",
            "GET /api/health": "Health check",
        },
    })
