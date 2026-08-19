import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from _lib.engine import reset_state

app = Flask(__name__)
application = app


@app.route("/api/reset", methods=["POST"])
def reset_user():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id") or request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id is required (in JSON body or query string)"}), 400
        return jsonify(reset_state(user_id))
    except Exception as e:
        logging.error(f"/api/reset error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
