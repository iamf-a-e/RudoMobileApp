import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from _lib.engine import health_status

app = Flask(__name__)
application = app


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        return jsonify(health_status())
    except Exception as e:
        logging.error(f"/api/health error: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
