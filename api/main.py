# Mobile-only Flask API (WhatsApp removed)

import os
import json
import re
import random
import string
import logging
from datetime import datetime
from .instructions import instructions
from .products_data import products_data
from .pregnancy_data import pregnancy_data
from .pregnancy_data_shona import pregnancy_data_shona
from .pregnancy_data_ndebele import pregnancy_data_ndebele
from .pregnancy_data_tonga import pregnancy_data_tonga
from .pregnancy_data_chinyanja import pregnancy_data_chinyanja
from .pregnancy_data_bemba import pregnancy_data_bemba
from .pregnancy_data_lozi import pregnancy_data_lozi
from .cervical_cancer_data import cervical_cancer_data
from flask import Flask, request, jsonify
import google.genai as genai
from upstash_redis import Redis

# =====================
# App & Logging
# =====================
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# =====================
# Environment & Config
# =====================
GEN_API = os.environ.get("GEN_API")
if not GEN_API:
    raise RuntimeError("GEN_API environment variable not set")

genai.configure(api_key=GEN_API)
MODEL_NAME = "gemini-2.0-flash"

# =====================
# Redis (Upstash)
# =====================
redis_client = None
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        redis_client.ping()
        logging.info("Connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Redis connection failed: {e}")
        redis_client = None

# =====================
# Global State (cached)
# =====================
user_states = {}

# =====================
# Helpers: State Persistence
# =====================

def load_user_states():
    global user_states
    if not redis_client:
        return
    try:
        data = redis_client.get("user_states")
        user_states = json.loads(data) if data else {}
    except Exception as e:
        logging.error(f"Failed to load states: {e}")
        user_states = {}


def save_user_states():
    if not redis_client:
        return
    try:
        redis_client.set("user_states", json.dumps(user_states))
    except Exception as e:
        logging.error(f"Failed to save states: {e}")

# =====================
# Language Detection
# =====================

def detect_language(text: str) -> str:
    shona_keywords = [
        "zviratidzo", "chiremba", "pamuviri", "kubuda",
        "ropa", "kurwadziwa", "mazamu"
    ]
    if any(word in text.lower() for word in shona_keywords):
        return "shona"
    return "english"

# =====================
# Gemini Helpers
# =====================

def ask_gemini_general(question: str, lang: str) -> str:
    try:
        if lang == "shona":
            instruction = (
                "Pindura mubvunzo uyu muShona yakapfava uye iri nyore kunzwisisa. "
                "Rangarira kuti izvi hazvitsivi kurairwa nachiremba:\n\n"
            )
        else:
            instruction = (
                "You are a professional health assistant specializing in maternal health "
                "and cervical cancer. Provide accurate, evidence-based information. "
                "End with a brief disclaimer that this does not replace a doctor's advice:\n\n"
            )

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(instruction + question)
        return response.text.strip() if response and response.text else (
            "Ndine urombo, handina kuwana mhinduro." if lang == "shona" else "Sorry, I couldn't find an answer."
        )
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "There was a problem answering your question."
        )

# =====================
# Core Conversation Logic (MOBILE)
# =====================

def handle_conversation_state_mobile(user_id: str, message: str) -> str:
    state = user_states[user_id]
    prompt = message.strip()
    prompt_lower = prompt.lower()

    # ---- Language detection ----
    if state["step"] == "language_detection":
        lang = detect_language(prompt)
        state["language"] = lang
        state["step"] = "registration"
        return (
            "Hello! Let's start with registration. What are the last 4 digits of your phone number?"
            if lang == "english"
            else "Mhoro! Ngatitangei nekunyoresa. Ndinokumbira manhamba mana ekupedzisira enhare yako."
        )

    # ---- Registration ----
    if state["step"] == "registration":
        digits = re.sub(r"\D", "", prompt)
        if len(digits) != 4:
            return (
                "Please enter exactly 4 digits." if state["language"] == "english"
                else "Ndapota pinda manhamba mana chete."
            )

        code = ''.join(random.choices(string.ascii_uppercase, k=4))
        state["user_code"] = f"DH-{digits}-{code}"
        state["registered"] = True
        state["step"] = "main_menu"

        return (
            f"Thank you! Your ID is {state['user_code']}. How can I help you today?\n- Maternal Health\n- Cervical Cancer"
            if state["language"] == "english"
            else f"Ndatenda! ID yako ndeye {state['user_code']}. Ndingakubatsirei nhasi?\n- Maternal Health\n- Cervical Cancer"
        )

    # ---- Main menu ----
    if state["step"] == "main_menu":
        if "maternal" in prompt_lower:
            state["topic"] = "maternal"
            state["step"] = "general_question"
            return (
                "You can now ask any maternal health question."
                if state["language"] == "english"
                else "Unogona kubvunza chero mubvunzo une chekuita nepamuviri."
            )

        if "cervical" in prompt_lower:
            state["topic"] = "cervical"
            state["step"] = "general_question"
            return (
                "You can now ask any cervical cancer question."
                if state["language"] == "english"
                else "Unogona kubvunza chero mubvunzo une chekuita ne gomarara rechibereko."
            )

        return (
            "Please choose one option:\n- Maternal Health\n- Cervical Cancer"
            if state["language"] == "english"
            else "Ndapota sarudza:\n- Maternal Health\n- Cervical Cancer"
        )

    # ---- General questions (Gemini) ----
    if state["step"] == "general_question":
        reply = ask_gemini_general(prompt, state["language"])
        return reply

    # ---- Fallback ----
    return ask_gemini_general(prompt, state.get("language", "english"))

# =====================
# API Routes
# =====================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"error": "user_id and message are required"}), 400

    load_user_states()

    if user_id not in user_states:
        user_states[user_id] = {
            "step": "language_detection",
            "language": "english",
            "registered": False
        }

    reply = handle_conversation_state_mobile(user_id, message)

    save_user_states()
    return jsonify({"reply": reply})


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        test_model = genai.GenerativeModel(MODEL_NAME)
        test_model.generate_content("Health check")

        return jsonify({
            "status": "healthy",
            "model": MODEL_NAME,
            "timestamp": str(datetime.now())
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    load_user_states()
    app.run(host="0.0.0.0", port=8000, debug=True)






