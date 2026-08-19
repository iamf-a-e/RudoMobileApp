"""
Dawa Health — Rudo Assistant
=============================
Full-featured WhatsApp bot (Cloud API webhook) + mobile-app JSON API,
merged into a single Flask `app` object so it can be deployed as a
Vercel Python serverless function (see api/index.py + vercel.json).

IMPORTANT — Vercel notes:
  * Serverless functions are stateless between invocations, so this file
    NEVER relies on the in-process `user_states` dict surviving across
    requests. Every read goes through `ensure_user_state()` /
    `load_user_state()`, which falls back to Redis (Upstash) on a cold
    start. Make sure UPSTASH_REDIS_URL / UPSTASH_REDIS_TOKEN are set,
    otherwise state will not persist between messages.
  * Vercel Hobby plan caps function execution at 10s (60s on Pro). Gemini
    calls are usually fast, but keep an eye on cold-start + model latency.
  * `training` and `products_data` are your own content modules (word
    lists, per-language pregnancy/cervical-cancer copy, product catalog,
    image URLs). They are imported defensively below — if they're missing
    the app still boots (so /api/health works) but bot replies will be
    degraded until you add them back into the deployment.
"""

import os
import re
import json
import random
import string
import logging
from datetime import datetime

from flask import Flask, request, jsonify, render_template
import requests
import google.generativeai as genai
from upstash_redis import Redis

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
#  CONTENT / DATA MODULES (defensive import)
# ─────────────────────────────────────────────
try:
    from training import instructions, product_images, products
    from training import (
        pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele,
        pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba,
        pregnancy_data_lozi, cervical_cancer_data, cervical_cancer_data_chinyanja,
        cervical_cancer_data_lozi,
    )
    try:
        from training import cervical_cancer_data_bemba, cervical_cancer_data_tonga
    except ImportError:
        cervical_cancer_data_bemba = cervical_cancer_data
        cervical_cancer_data_tonga = cervical_cancer_data
    from products_data import products_by_category
    CONTENT_LOADED = True
    logging.info("Content modules (training/, products_data.py) loaded OK.")
except Exception as e:  # pragma: no cover - defensive only
    logging.error(f"Content modules missing/broken ({e}). Bot will run in degraded mode.")
    CONTENT_LOADED = False

    class _EmptyModule:
        def __getattr__(self, _):
            return ""

    instructions = product_images = products = _EmptyModule()
    pregnancy_data = pregnancy_data_shona = pregnancy_data_ndebele = _EmptyModule()
    pregnancy_data_tonga = pregnancy_data_chinyanja = pregnancy_data_bemba = _EmptyModule()
    pregnancy_data_lozi = cervical_cancer_data = cervical_cancer_data_chinyanja = _EmptyModule()
    cervical_cancer_data_lozi = cervical_cancer_data_bemba = cervical_cancer_data_tonga = _EmptyModule()
    products_by_category = {}
    product_images.image_urls = {}
    pregnancy_data.pregnancy_data = ""
    pregnancy_data_shona.pregnancy_data_shona = ""
    pregnancy_data_ndebele.pregnancy_data_ndebele = ""
    pregnancy_data_chinyanja.pregnancy_data_chinyanja = ""
    pregnancy_data_bemba.pregnancy_data_bemba = ""
    pregnancy_data_lozi.pregnancy_data_lozi = ""
    pregnancy_data_tonga.pregnancy_data_tonga = ""
    cervical_cancer_data.cervical_cancer_data = []
    cervical_cancer_data_chinyanja.cervical_cancer_data_chinyanja = []
    cervical_cancer_data_lozi.cervical_cancer_data_lozi = []
    cervical_cancer_data_bemba.cervical_cancer_data_bemba = []
    cervical_cancer_data_tonga.cervical_cancer_data_tonga = []

# ─────────────────────────────────────────────
#  OPTIONAL: Firebase / Firestore (from the mobile-app build)
#  Used only for a secondary, best-effort chat-history log for the
#  mobile app endpoints. Redis remains the source of truth for bot state.
# ─────────────────────────────────────────────
firestore_db = None
try:
    from firebase_admin import credentials, firestore, initialize_app
    import firebase_admin

    firebase_creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if firebase_creds:
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(firebase_creds))
            initialize_app(cred)
        firestore_db = firestore.client()
        logging.info("Firebase Firestore initialized")
    else:
        logging.info("FIREBASE_SERVICE_ACCOUNT not set — Firestore chat-history log disabled.")
except Exception as e:
    logging.warning(f"Firebase not available ({e}) — continuing without Firestore.")
    firestore_db = None


def save_message_to_firestore(user_id: str, message: str, is_user: bool = True):
    if not firestore_db:
        return
    try:
        chat_ref = firestore_db.collection("whatsapp_chats").document(user_id)
        chat_ref.set({
            "user_id": user_id,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "last_message": message[:100],
        }, merge=True)
        chat_ref.collection("messages").add({
            "content": message,
            "is_user": is_user,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "created_at": datetime.now().isoformat(),
        })
    except Exception as e:
        logging.error(f"Failed to save to Firestore: {e}")


def get_chat_history_firestore(user_id: str, limit: int = 50):
    if not firestore_db:
        return []
    try:
        messages_ref = firestore_db.collection("whatsapp_chats").document(user_id).collection("messages")
        docs = messages_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                "id": doc.id,
                "content": data.get("content", ""),
                "is_user": data.get("is_user", False),
                "timestamp": str(data.get("timestamp", "")),
                "created_at": data.get("created_at", ""),
            })
        messages.reverse()
        return messages
    except Exception as e:
        logging.error(f"Failed to get chat history: {e}")
        return []


# ─────────────────────────────────────────────
#  Upstash Redis (state + conversation store)
# ─────────────────────────────────────────────
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN not set — state will NOT persist "
                     "between serverless invocations. Set these env vars in Vercel.")

# Per-invocation cache only — never assume this survives between requests on Vercel.
user_states = {}

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.5-flash"

if gen_api:
    genai.configure(api_key=gen_api)
else:
    logging.error("GEN_API environment variable not set — Gemini calls will fail.")

name = "Fae"
bot_name = "Rudo"

# ── AGENT DICTIONARY ─────────────────────────────────────────────────────────
AGENTS = {
    "Agent 1": os.environ.get("AGENT_1_PHONE", "+260978760105"),
}
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TIMEOUT_SECONDS = 60  # seconds before "no agents available" fires

app = Flask(__name__)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


# ─────────────────────────────────────────────
#  SHARED MATCHING HELPER
# ─────────────────────────────────────────────

def _contains_signal(prompt_lower: str, phrases: list) -> bool:
    for phrase in phrases:
        phrase_l = phrase.lower()
        if " " in phrase_l:
            if phrase_l in prompt_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(phrase_l)}\b", prompt_lower):
                return True
    return False


# ─────────────────────────────────────────────
#  PER-USER REDIS STATE
# ─────────────────────────────────────────────

def save_single_user_state(sender):
    if redis_client and sender in user_states:
        try:
            redis_client.set(f"user_state:{sender}", json.dumps(user_states[sender]))
        except Exception as e:
            logging.error(f"Error saving state for {sender}: {e}")


def load_user_state(sender):
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            if state_data:
                return json.loads(state_data)
        except Exception as e:
            logging.error(f"Error loading user state for {sender}: {e}")
    return None


def save_user_states():
    for sender in list(user_states.keys()):
        save_single_user_state(sender)


def load_user_states():
    global user_states
    user_states = {}
    logging.info("User states initialised (lazy per-user loading enabled)")


def ensure_user_state(sender):
    if sender in user_states:
        return False

    saved = load_user_state(sender)
    if saved:
        user_states[sender] = saved
        return False

    user_states[sender] = {
        "step": "language_detection",
        "language": "english",
        "registered": False,
        "phone_digits": None,
        "user_id": None,
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": True,
    }
    return True


def reset_conversation(sender):
    user_states[sender] = {
        "step": "main_menu",
        "language": user_states[sender].get("language", "english"),
        "registered": True,
        "phone_digits": user_states[sender].get("phone_digits"),
        "user_id": user_states[sender].get("user_id"),
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": False,
    }
    save_single_user_state(sender)


# ─────────────────────────────────────────────
#  HUMAN AGENT SYSTEM
# ─────────────────────────────────────────────

HUMAN_AGENT_TRIGGERS = [
    "human agent", "speak to agent", "talk to agent", "talk to a person", "human",
    "speak to a person", "real person", "human support", "live agent", "support",
    "live support", "connect me to an agent", "i need help from a person",
    "speak to someone", "talk to someone", "customer service", "agent",
]


def is_human_agent_request(prompt: str) -> bool:
    p = prompt.lower().strip()
    return _contains_signal(p, HUMAN_AGENT_TRIGGERS)


def normalize_phone(phone: str) -> str:
    return phone.lstrip("+")


def send_interactive_buttons(phone_number: str, body_text: str, buttons: list, phone_id_val: str):
    url = f"https://graph.facebook.com/v19.0/{phone_id_val}/messages"
    headers = {
        "Authorization": f"Bearer {wa_token}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": btn["id"], "title": btn["title"]}}
                    for btn in buttons
                ]
            },
        },
    }
    resp = requests.post(url, headers=headers, json=data, timeout=15)
    logging.info(f"Interactive button send to {phone_number}: {resp.status_code} {resp.text}")
    return resp


def _agent_request_key(user_number: str) -> str:
    return f"agent_request:{user_number}"


def _agent_session_key(user_number: str) -> str:
    return f"agent_session:{user_number}"


def _agent_rejections_key(user_number: str) -> str:
    return f"agent_rejections:{user_number}"


def notify_agents_of_request(sender: str, current_phone_id: str):
    state = user_states.get(sender, {})
    user_id = state.get("user_id", sender)
    lang = state.get("language", "english")

    request_data = {
        "user_number": sender,
        "user_id": user_id,
        "language": lang,
        "phone_id": current_phone_id,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "accepted_by": None,
        "rejections": [],
    }

    if redis_client:
        try:
            redis_client.set(_agent_request_key(sender), json.dumps(request_data), ex=AGENT_TIMEOUT_SECONDS + 10)
            redis_client.set(_agent_rejections_key(sender), json.dumps([]), ex=AGENT_TIMEOUT_SECONDS + 10)
        except Exception as e:
            logging.error(f"Error saving agent request to Redis: {e}")

    body = (
        f"🔔 *New Chat Request*\n\n"
        f"User ID : {user_id}\n"
        f"Language: {lang}\n"
        f"Phone   : {sender}\n\n"
        f"Do you want to accept this chat?"
    )
    buttons = [
        {"id": f"agent_accept:{sender}", "title": "✅ Accept"},
        {"id": f"agent_reject:{sender}", "title": "❌ Reject"},
    ]

    for agent_name, agent_phone in AGENTS.items():
        logging.info(f"Notifying agent {agent_name} ({agent_phone}) of request from {sender}")
        send_interactive_buttons(agent_phone, body, buttons, current_phone_id)


def handle_agent_accept(agent_phone: str, user_number: str, current_phone_id: str):
    if not redis_client:
        send("Sorry, the agent system is unavailable right now.", user_number, current_phone_id)
        return
    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send_interactive_buttons(
                agent_phone,
                "⚠️ This chat request has already expired or been accepted by another agent.",
                [], current_phone_id,
            )
            send("⚠️ This chat request has already expired or been accepted by another agent.", agent_phone, current_phone_id)
            return

        request_data = json.loads(raw)
        if request_data.get("status") != "pending":
            send("⚠️ This chat has already been accepted by another agent.", agent_phone, current_phone_id)
            return

        request_data["status"] = "accepted"
        request_data["accepted_by"] = agent_phone
        redis_client.set(_agent_request_key(user_number), json.dumps(request_data), ex=3600)

        session_data = {
            "user_number": user_number,
            "agent_phone": agent_phone,
            "phone_id": current_phone_id,
            "started_at": datetime.now().isoformat(),
        }
        redis_client.set(_agent_session_key(user_number), json.dumps(session_data), ex=3600)
        redis_client.set(f"agent_user_session:{normalize_phone(agent_phone)}", json.dumps(session_data), ex=3600)

        agent_name = next((n for n, p in AGENTS.items() if p == agent_phone), agent_phone)

        ensure_user_state(user_number)
        user_states[user_number]["step"] = "human_agent_chat"
        user_states[user_number]["agent_phone"] = agent_phone
        save_single_user_state(user_number)

        send(
            f"✅ You are now connected to the user ({user_number}).\n"
            f"Their messages will be forwarded to you. Reply here to send messages to them.\n"
            f"Type *END CHAT* to end the session.",
            agent_phone, current_phone_id,
        )
        send(
            f"✅ Great news! {agent_name} has accepted your chat request.\n"
            f"You are now connected!",
            user_number, current_phone_id,
        )
        for other_name, other_phone in AGENTS.items():
            if other_phone != agent_phone:
                send(f"ℹ️ The chat request from user {user_number} has been accepted by {agent_name}.", other_phone, current_phone_id)

    except Exception as e:
        logging.error(f"Error in handle_agent_accept: {e}", exc_info=True)


def handle_agent_reject(agent_phone: str, user_number: str, current_phone_id: str):
    if not redis_client:
        return
    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send("⚠️ This chat request has already expired.", agent_phone, current_phone_id)
            return

        request_data = json.loads(raw)
        if request_data.get("status") != "pending":
            send("ℹ️ This request was already handled.", agent_phone, current_phone_id)
            return

        rejections_raw = redis_client.get(_agent_rejections_key(user_number))
        rejections = json.loads(rejections_raw) if rejections_raw else []
        if agent_phone not in rejections:
            rejections.append(agent_phone)
        redis_client.set(_agent_rejections_key(user_number), json.dumps(rejections), ex=AGENT_TIMEOUT_SECONDS + 10)

        send("👍 You have rejected this chat request.", agent_phone, current_phone_id)

        if set(rejections) >= set(AGENTS.values()):
            _handle_no_agents_available(user_number, current_phone_id)

    except Exception as e:
        logging.error(f"Error in handle_agent_reject: {e}", exc_info=True)


def _handle_no_agents_available(user_number: str, current_phone_id: str):
    ensure_user_state(user_number)
    user_states[user_number]["step"] = "main_menu"
    user_states[user_number].pop("agent_phone", None)
    save_single_user_state(user_number)

    if redis_client:
        try:
            redis_client.delete(_agent_request_key(user_number))
            redis_client.delete(_agent_rejections_key(user_number))
        except Exception:
            pass

    lang = user_states[user_number].get("language", "english")
    no_agent_map = {
        "shona": "😔 Ndine urombo, hapana mubatsiri anowanikwa parizvino.\nNdinokudzoseredzai kuna Rudo, mubatsiri wedu wepamhepo.\nPane chimwe chandingakubatsira nacho here?",
        "ndebele": "😔 Uxolo, awukho umuntu otholakalayo okwamanje.\nSiyakubuyisela ku-Rudo, umsizi wethu we-inthanethi.\nIngabe kukhona okunye engingakusiza ngakho?",
        "chinyanja": "😔 Pepani, palibe wothandiza ali ndi ntchito pakali pano.\nTikubwereza kwa Rudo, wothandiza wathu wa intaneti.\nKodi pali zina zomwe ndingakuthandizireni?",
        "bemba": "😔 Njelelako, tapali mwafwilishi ulipo pali ino nshita.\nTulakulekela kuli Rudo, umwafwilishi wesu wa ku intaneti.\nKuli fintu fimbi ifyo ningamwafwilisha?",
        "lozi": "😔 Ni maswabi, ha ku na mubasi ya fumaneha cwale.\nLu ku kutiseza kwa Rudo, mubasi wa luna wa intaneti.\nKi sina sika ni ka thusa ka sona?",
        "tonga": "😔 Ndatola, kunyina wakugwasya ulikonzeka lino.\nTulamubweza kuli Rudo, wakugwasya wesu wa intaneti.\nHena muli amubuyo umbi?",
    }
    send(no_agent_map.get(lang, "😔 Sorry, no agents are available right now.\nYou have been handed back to Rudo, our virtual assistant.\nIs there anything else I can help you with?"), user_number, current_phone_id)


def relay_user_message_to_agent(sender: str, prompt: str, current_phone_id: str) -> bool:
    if not redis_client:
        return False
    try:
        raw = redis_client.get(_agent_session_key(sender))
        if not raw:
            return False
        session = json.loads(raw)
        agent_phone = session.get("agent_phone")
        if not agent_phone:
            return False
        send(f"💬 *User ({sender}):* {prompt}", agent_phone, current_phone_id)
        return True
    except Exception as e:
        logging.error(f"Error relaying user message to agent: {e}")
        return False


def relay_agent_message_to_user(agent_phone: str, prompt: str, current_phone_id: str) -> bool:
    if not redis_client:
        return False
    try:
        norm_agent = normalize_phone(agent_phone)
        raw = redis_client.get(f"agent_user_session:{norm_agent}")
        if not raw:
            return False
        session = json.loads(raw)
        user_number = session.get("user_number")
        if not user_number:
            return False

        prompt_stripped = prompt.strip()

        if prompt_stripped.upper() == "END CHAT":
            redis_client.delete(f"agent_user_session:{norm_agent}")
            redis_client.delete(_agent_session_key(user_number))
            redis_client.delete(_agent_request_key(user_number))

            ensure_user_state(user_number)
            user_states[user_number]["step"] = "main_menu"
            user_states[user_number].pop("agent_phone", None)
            save_single_user_state(user_number)

            send("✅ You have ended the chat session.", agent_phone, current_phone_id)
            lang = user_states[user_number].get("language", "english")
            end_map = {
                "shona": "👋 Mubatsiri wangu akunge ngumi chisarai.\nMadzoka kumubatsiri wedu wepamhepo Rudo.\nPane chimwe chandingakubatsira nacho?",
                "ndebele": "👋 Umhloli wami uphethile inkulumo yakho.\nUbuyela ku-Rudo, umsizi wethu we-inthanethi.\nIngabe kukhona okunye engingakusiza ngakho?",
                "chinyanja": "👋 Wothandiza wanu wamaliza kuchatitana nawo.\nMumabwerera kwa Rudo, wothandiza wathu wa intaneti.\nKodi pali zina zomwe ndingakuthandizireni?",
                "bemba": "👋 Umwafwilishi wenu ashile ukumana nawe.\nMwabwela kuli Rudo, umwafwilishi wesu wa ku intaneti.\nKuli fintu fimbi ifyo ningamwafwilisha?",
                "lozi": "👋 Mubasi wa hao u felelize puisano ya hao.\nU bwela kwa Rudo, mubasi wa luna wa intaneti.\nKi sina sika ni ka thusa ka sona?",
                "tonga": "👋 Wakugwasya wanu wamanizya kulumbaana anywi.\nMubweza kuli Rudo, wakugwasya wesu wa intaneti.\nHena muli amubuyo umbi?",
            }
            send(end_map.get(lang, "👋 Your agent has ended the chat session.\nYou have been returned to Rudo, our virtual assistant.\nIs there anything else I can help you with?"), user_number, current_phone_id)
            return True

        send(f"💬 *Agent:* {prompt_stripped}", user_number, current_phone_id)
        return True

    except Exception as e:
        logging.error(f"Error relaying agent message to user: {e}")
        return False


def check_agent_request_timeout(user_number: str, current_phone_id: str):
    if not redis_client:
        _handle_no_agents_available(user_number, current_phone_id)
        return
    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            _handle_no_agents_available(user_number, current_phone_id)
    except Exception as e:
        logging.error(f"Error checking agent timeout: {e}")
        _handle_no_agents_available(user_number, current_phone_id)


# ─────────────────────────────────────────────
#  REFERRAL SOURCE TRACKING
# ─────────────────────────────────────────────

REFERRAL_TRIGGERS = [
    "got connected from", "i got connected from", "i was connected from",
    "connected from", "referred from", "coming from", "came from",
    "i came from", "i'm from", "im from", "from the poster",
    "from a poster", "from flyer", "from a flyer", "from the flyer",
    "saw your poster", "saw a poster", "saw the poster",
    "hey dawamom", "hi dawamom", "hello dawamom",
]


def extract_referral_source(prompt: str):
    prompt_lower = prompt.lower().strip()
    triggered = any(t in prompt_lower for t in REFERRAL_TRIGGERS)
    if not triggered:
        return None
    for prep in ["from ", "from the ", "from a "]:
        idx = prompt_lower.rfind(prep)
        if idx != -1:
            source = prompt[idx + len(prep):].strip().rstrip(".,!?;:")
            if source:
                return source
    for trigger in sorted(REFERRAL_TRIGGERS, key=len, reverse=True):
        idx = prompt_lower.find(trigger)
        if idx != -1:
            source = prompt[idx + len(trigger):].strip().rstrip(".,!?;:")
            if source:
                return source
    return None


def save_referral_source(sender: str, source: str):
    if not redis_client:
        return
    try:
        entry = {
            "sender": sender,
            "source": source,
            "user_id": user_states.get(sender, {}).get("user_id", "unregistered"),
            "timestamp": datetime.now().isoformat(),
        }
        key = f"referrals:{sender}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        redis_client.set(key, json.dumps(entry))
    except Exception as e:
        logging.error(f"Error saving referral: {e}")


def get_user_conversation(sender):
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            if not history:
                return []
            if isinstance(history, list):
                return history
            if isinstance(history, str):
                parsed = json.loads(history)
                if isinstance(parsed, list):
                    return parsed
            return []
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return []
    return []


def save_user_conversation(sender, role, message):
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            if not isinstance(conversation, list):
                conversation = []
            conversation.append({"role": role, "message": str(message), "timestamp": datetime.now().isoformat()})
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.set(f"conversation:{sender}", json.dumps(conversation), ex=60 * 60 * 24 * 30)
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")
    # best-effort secondary log
    save_message_to_firestore(sender, message, is_user=(role == "user"))


# ─────────────────────────────────────────────
#  LLM FALLBACK LANGUAGE CLASSIFIER
# ─────────────────────────────────────────────

def _llm_detect_language(message: str):
    supported = ["english", "shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"]
    try:
        classifier_prompt = (
            "Identify which ONE of these languages the following WhatsApp message "
            "is written in: english, shona, ndebele, chinyanja, bemba, tonga, lozi. "
            "These are languages spoken in Zimbabwe and Zambia. The message may mix "
            "in a few English loanwords (e.g. medical terms like 'cervical cancer' "
            "or 'HPV') while still being primarily one of the other languages — in "
            "that case, classify by the surrounding grammar/vocabulary, not the "
            "loanwords. Reply with ONLY the single lowercase language name and "
            "nothing else — no punctuation, no explanation.\n\n"
            f"Message: \"{message}\""
        )
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": 0, "max_output_tokens": 10},
            safety_settings=safety_settings,
        )
        response = gemini_model.generate_content(classifier_prompt)
        guess = re.sub(r"[^a-z]", "", response.text.strip().lower())
        if guess in supported:
            return guess
    except Exception as e:
        logging.error(f"[_llm_detect_language Error] {type(e).__name__}: {e}")
    return None


def detect_language(message, sender=None):
    message_lower = message.lower().strip()

    if message_lower.isdigit():
        if sender and sender in user_states:
            return user_states[sender].get("language", "english")
        return "english"

    current_lang = "english"
    if sender and sender in user_states:
        current_lang = user_states[sender].get("language", "english")

    exact_matches = {
        "shona":     ["mhoro", "mhoroi", "makadini", "hesi", "hapana", "ndizvo", "zvakanaka", "wadini", "taura", "kwete"],
        "ndebele":   ["sawubona", "salibonani", "unjani", "yebo", "ngiyabonga", "ngicela", "impela", "kunjani", "hatshi", "kambe"],
        "bemba":     ["mwaiseni", "ulishani", "nalikutemwa", "natotela", "shani", "chisuma", "sana", "njelelako", "twatotela", "mukwai", "napapata"],
        "chinyanja": ["moni", "zikomo", "pepani", "ndithu", "chonde", "eyaa", "nitandizeni", "nankani"],
        "tonga":     ["mwabuka", "mwalandwa", "ndatotela", "kapati", "mbuti"],
        "lozi":      ["ndalumba", "haa", "kacenu", "muzuhile"],
    }
    for lang, words in exact_matches.items():
        if message_lower in words:
            return lang

    language_keywords = {
        "shona": ["mhoro", "mhoroi", "makadini", "ndinonzi", "zvakanaka", "ndatenda", "pamuviri", "zvigadzirwa",
                  "chirwere", "gomarara", "chibereko", "zviratidzo", "chiremba", "kusvotwa", "kurwadziwa",
                  "handina", "ndinoda", "zvichava", "zvakadaro", "kwete", "hapana", "ndizvo", "zvakafanana",
                  "ndoziva", "nhumbu", "ndine", "ndiri", "ndinoziva", "sei", "zvii", "vanhu", "muviri", "mazuva",
                  "hesi", "masvingo", "musha", "kuita", "ndakadaro", "zviripo", "zvinobvira"],
        "ndebele": ["sawubona", "salibonani", "unjani", "ngiyabonga", "ngicela", "isisu", "umntwana", "imikhiqizo",
                    "umhlaza", "isibeletho", "izimpawu", "udokotela", "igazi", "ubuhlungu", "angikwazi", "ngifuna",
                    "ukukhulelwa", "abantu", "akukho", "impela", "kakhulu"],
        "chinyanja": ["moni", "zikomo", "pepani", "ndapota", "matenda", "kansa", "zizindikiro", "dokotala",
                      "magazi", "zabwino", "sindikudziwa", "ndikufuna", "sabata", "zambiri", "thanzo", "mavitamini",
                      "nitandizeni", "nankani", "vumo", "mimba", "bwanji", "thandizani", "ndimva", "ndikumva", "ndinafuna"],
        "lozi": ["ndalumba", "zibonelelo", "kuhula", "kushisa", "maviki", "mutango", "mupilo", "mubonelelo",
                 "kacenu", "muzuhile", "kimanzibwana", "mulumele", "mutu", "lilimo", "silelezwa", "butuku",
                 "musimbi", "bulwazi", "cwale", "wakona", "wapimwa"],
        "bemba": ["mwaiseni", "nalikutemwa", "natotela", "twatotela", "mukwai", "ngafweniko", "cilikwisa",
                  "ubushiku", "ifyakulya", "ukubomba", "icisungu", "icibemba", "shaleenipo"],
        "tonga": ["ndalumba", "lugwazyo", "mubuzyo", "kapati", "zitondezyo", "mutumbu", "dokota", "cinzi",
                  "buti", "makani", "kusilikwa", "mbubo", "buumi", "chibadela", "kaambo nzi"],
        "english": ["what", "how", "when", "why", "where", "signs", "symptoms", "information", "please",
                    "thank", "sorry", "help", "watch", "during", "risky"],
    }
    language_phrases = {
        "chinyanja": ["muli bwanji", "uli ndi chani", "zikomo kwambiri", "muli bwino", "ndili bwino", "nitandizeni nankani"],
        "shona":     ["makadini", "zvakanaka sei", "ndatenda", "ndiriku"],
        "ndebele":   ["unjani wena", "ngiyabonga kakhulu", "sicela ungichazele"],
        "lozi":      ["uli bwanji", "ni bata", "ha ndi zibi", "ndalumba hahulu"],
        "bemba":     ["muli shani", "napapata", "nshishibe", "bushe kuti"],
        "tonga":     ["mmuli buti", "ndakomba", "sena kuti"],
        "english":   ["how are you", "what are", "what is", "can you", "tell me", "i need", "i want", "please tell",
                      "watch out", "how do i", "how can i", "what should"],
    }

    scores = {lang: 0 for lang in language_keywords}
    for lang, phrases in language_phrases.items():
        for phrase in phrases:
            if phrase in message_lower:
                scores[lang] = scores.get(lang, 0) + 5
    for lang, keywords in language_keywords.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", message_lower):
                scores[lang] = scores.get(lang, 0) + 3

    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        top_langs = [lang for lang, s in scores.items() if s == max_score]
        if current_lang in top_langs:
            return current_lang
        if len(top_langs) == 1 and max_score >= 5:
            return top_langs[0]
        return current_lang

    words_in_msg = re.findall(r"[a-z]+", message_lower)
    if len(words_in_msg) >= 3:
        llm_guess = _llm_detect_language(message)
        if llm_guess:
            return llm_guess

    common_english_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall", "can", "need", "must", "ought",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "its", "our", "their", "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "and", "or", "but", "if", "then", "so", "because", "although", "while",
        "not", "no", "yes", "please", "thank", "thanks", "sorry", "okay", "ok", "to", "of", "in", "on", "at",
        "for", "from", "with", "about", "during", "tell", "give", "show", "help", "know", "want", "need",
        "get", "go", "come", "see", "look", "take", "make", "say", "ask", "work", "feel", "think", "try",
        "use", "find", "early", "late", "common", "normal", "severe", "pain", "blood", "baby", "mother",
        "health", "information", "more", "other", "any", "all", "some", "much", "many", "very", "also",
        "just", "only", "still", "even", "back", "too", "well", "good", "bad", "new", "old", "long", "little",
        "right", "big", "high", "low", "next", "last", "between", "after", "before", "since", "until",
        "without", "within", "up", "down", "over", "under", "again", "further", "once", "same", "own",
        "both", "each", "few", "most", "such", "than", "as", "by", "into", "through", "against", "along",
        "following", "across", "behind", "beyond", "plus", "except", "including", "throughout", "towards",
        "upon", "concerning",
    }
    unique_words = set(words_in_msg)
    if len(words_in_msg) >= 5 and unique_words:
        en_count = sum(1 for w in unique_words if w in common_english_words)
        ratio = en_count / len(unique_words)
        if ratio >= 0.40:
            if current_lang != "english" and ratio < 0.7:
                return current_lang
            return "english"

    if all(ord(c) < 128 for c in message_lower):
        if current_lang != "english":
            return current_lang
        return "english"

    return "english"


def get_pregnancy_data(language):
    return {
        "shona": pregnancy_data_shona.pregnancy_data_shona,
        "ndebele": pregnancy_data_ndebele.pregnancy_data_ndebele,
        "chinyanja": pregnancy_data_chinyanja.pregnancy_data_chinyanja,
        "lozi": pregnancy_data_lozi.pregnancy_data_lozi,
        "bemba": pregnancy_data_bemba.pregnancy_data_bemba,
        "tonga": pregnancy_data_tonga.pregnancy_data_tonga,
    }.get(language, pregnancy_data.pregnancy_data)


def get_cervical_data(language):
    return {
        "shona": cervical_cancer_data.cervical_cancer_data,
        "ndebele": cervical_cancer_data.cervical_cancer_data,
        "chinyanja": cervical_cancer_data_chinyanja.cervical_cancer_data_chinyanja,
        "lozi": cervical_cancer_data_lozi.cervical_cancer_data_lozi,
        "bemba": cervical_cancer_data_bemba.cervical_cancer_data_bemba,
        "tonga": cervical_cancer_data_tonga.cervical_cancer_data_tonga,
    }.get(language, cervical_cancer_data.cervical_cancer_data)


def send(answer, sender, phone_id_):
    url = f"https://graph.facebook.com/v19.0/{phone_id_}/messages"
    headers = {"Authorization": f"Bearer {wa_token}", "Content-Type": "application/json"}
    type_ = "text"
    body = "body"
    content = answer
    image_urls = getattr(product_images, "image_urls", {})

    if "product_image" in answer:
        product_match = re.search(r"product_image_(\w+)", answer)
        if product_match:
            product_name = product_match.group(1)
            if product_name in image_urls:
                image_url = image_urls[product_name]
                from mimetypes import guess_type
                mime_type, _ = guess_type(image_url.split("/")[-1])
                if mime_type and mime_type.startswith("image"):
                    type_ = "image"
                    body = "link"
                    content = image_url
                    answer = re.sub(r"product_image_\w+", "", answer)

    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": type_,
        type_: {body: content, **({"caption": answer.strip()} if type_ != "text" else {})},
    }

    if not wa_token or not phone_id_:
        logging.warning("WA_TOKEN/phone_id not configured — skipping outbound WhatsApp send.")
        save_user_conversation(sender, "bot", answer)
        return None

    response = requests.post(url, headers=headers, json=data, timeout=15)
    logging.info(f"Send status: {response.status_code} {response.text}")
    save_user_conversation(sender, "bot", answer)
    return response


# ─────────────────────────────────────────────
#  Continuous language detection
# ─────────────────────────────────────────────

def maybe_update_language(sender, prompt):
    state = user_states[sender]
    current_step = state.get("step", "main_menu")
    if current_step in ["language_detection", "registration"]:
        return state.get("language", "english")
    if prompt.strip().isdigit():
        return state.get("language", "english")
    detected = detect_language(prompt, sender)
    current_lang = state.get("language", "english")
    if detected != current_lang:
        state["language"] = detected
        save_single_user_state(sender)
    return state["language"]


def handle_language_detection(sender, prompt, phone_id_):
    detected_lang = detect_language(prompt, sender)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "registration"
    user_states[sender]["needs_language_confirmation"] = False

    greetings = {
        "shona": "Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.",
        "ndebele": "Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.",
        "bemba": "Mwaiseni! Nine Rudo, wakufwailisha wa Dawa Health. Tiyeni tampilepo ukulembesha. Cisuma mpeele amanamba ayi 4 ayalekelesha sha ku foni namba yenu.",
        "chinyanja": "Moni! Ndine Rudo, mphungu wa Dawa Health. Tiyambireni ndi kulembetsa. Chonde ndipatseni manambala anayi omaliza a nambala yanu yafoni.",
        "tonga": "Muli buti! Ndime Rudo, wakugwasya Dawa Health. Atutalikile kulembezya. amundipe ma nambala ali 4 ali kumamanino ya foni namba yenu",
        "lozi": "Mwa bona! Mina ki Rudo, mubasi wa ku thusa wa Dawa Health wa ku kompyuta. A re simule ka ku itambula. Ndapota, nipe dinomolo za mafelele a mane za foni ya hao.",
    }
    send(greetings.get(detected_lang, "Hello! I'm Rudo, Dawa Health's virtual assistant. Let's start with registration. Please tell me the last 4 digits of your phone number."), sender, phone_id_)
    save_single_user_state(sender)


def handle_registration(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_clean = prompt.strip()

    if state.get("phone_digits") is None:
        if not re.fullmatch(r"\d{4}", prompt_clean):
            invalid_map = {
                "shona": "Ndapota nyorai manhamba mana chete ekupedzisira enhare yenyu (semuenzaniso: 1234).",
                "ndebele": "Ngicela ubhale amadijithi amane kuphela okugcina enombolweni yakho yocingo (isibonelo: 1234).",
                "chinyanja": "Chonde lembani manambala anayi okha omaliza a nambala yanu yafoni (mwachitsanzo: 1234).",
                "tonga": "Ndakomba mulembe ma nambala aane luzutu aakumaninina anambala yenu ya foni (mucikozyanyo: 1234).",
                "bemba": "Napapata lembeni fye amanambala 4 ayakulekelesha kuli nambala yenu ya foni (ichilangililo: 1234).",
                "lozi": "Ndapota ñola dinomolo za mafelele a mane feela za foni ya hao (mutala: 1234).",
            }
            send(invalid_map.get(lang, "Please send only the last 4 digits of your phone number (e.g. 1234)."), sender, phone_id_)
            save_single_user_state(sender)
            return

        state["phone_digits"] = prompt_clean
        random_letters = "".join(random.choices(string.ascii_uppercase, k=4))
        user_id = f"DH-{prompt_clean}-{random_letters}"
        state["user_id"] = user_id

        welcome_map = {
            "shona": f"Ndatenda! ID yenyu yakagadzirwa ndeye: {user_id}. Chengetedza ID iyi nekuti ichakumbirwa kumaDawa clinics. Ndingakubatsirei nhasi?",
            "ndebele": f"Ngiyabonga! I-ID yakho eyakhiwe ithi: {user_id}. Gcina le ID ngoba izocelwa kumaDawa clinics. Ngingakusiza ngani namuhla?",
            "bemba": f"Natotela! ID yenu iyapangwa ni: {user_id}. Sungeni ID iyi pantu ikabombwa ku Dawa clinics. Nga kuti namwafwa shani lelo?",
            "chinyanja": f"Zikomo! ID yanu yopangidwa ndi: {user_id}. Sungani ID iyi chifukwa idzafunsidwa kumakliniki a Dawa. Ndingakuthandizireni lero?",
            "tonga": f"Twalumba! ID yenu nji: {user_id}. mweelede kuisunga kabotu ID kambo iyakubeleka ku Dawa clinics. Nga ndamukyasya buti lino?",
            "lozi": f"Ndalumba! ID ya wena ye e bupilwe ki: {user_id}. Boloka ID ye hantši kakuli u ta buzwa yona kwa makiliniki a Dawa. Nka ku thusa ka mini sunu?",
        }
        send(welcome_map.get(lang, f"Thank you! Your generated ID is: {user_id}. Keep this ID safe because it'll be asked for at the Dawa clinics. How can I help you today?"), sender, phone_id_)

        state["registered"] = True
        state["step"] = "main_menu"

    save_single_user_state(sender)


def is_exact_match(text, responses):
    words = re.findall(r"\b\w+\b", text)
    return any(word in responses for word in words)


def _send_thinking(sender, phone_id_, lang):
    thinking_map = {
        "shona": "Ndiri kufunga...", "ndebele": "Ngiyacabangisisa...", "chinyanja": "Ndikuganiza...",
        "tonga": "Ndichiyandaula...", "bemba": "ndefwailisha...", "lozi": "Ni nahana...",
    }
    send(thinking_map.get(lang, "Let me think..."), sender, phone_id_)


def _send_more_questions(sender, phone_id_, lang):
    more_map = {
        "shona": "Pane chimwe chamunoda kubvunza here?", "ndebele": "Uneminye imibuzo yini?",
        "chinyanja": "Kodi muli ndi mafunso ena?", "tonga": "Hena muli a mubuzyo?",
        "bemba": "Uli ne fipusho nafimbi?", "lozi": "O na mabvuzo a mangi?",
    }
    send(more_map.get(lang, "Do you have any more questions?"), sender, phone_id_)


GREETING_WORDS = [
    "hi", "hello", "hey", "hie", "hi there", "good morning", "good afternoon", "good evening",
    "mhoro", "mhoroi", "hesi", "makadini", "wadini", "sawubona", "salibonani", "moni", "muli bwanji",
    "mwabuka", "mwabuka buti", "mwatambulwa", "buti", "mwaiseni", "muli shani", "mwa bona",
    "mbuti", "mwalandwa", "mwalandwa buti",
]
RESET_KEYWORDS = ["start over", "restart", "new conversation", "main menu", "menu", "reset", "help"]
GREET_MAP = {
    "shona": "Mhoroi! Ndingakubatsirei nhasi?", "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
    "chinyanja": "Moni! Ndingakuthandizireni lero?", "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
    "tonga": "Muli buti! Nga ndamukwasya buti sunu?", "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
}
YES_RESPONSES = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo", "emukwayi"]
NO_RESPONSES = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "awe", "pepe", "cha", "ayi", "iyo"]


def handle_follow_up(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, GREETING_WORDS):
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_)
        save_single_user_state(sender)
        return

    if _contains_signal(prompt_lower, NO_RESPONSES):
        _ask_purchase_interest(sender, phone_id_, lang)
        return

    if len(prompt_lower.split()) > 2:
        _send_thinking(sender, phone_id_, lang)

    reply = ask_gemini_general(prompt, lang, sender=sender)
    send(reply, sender, phone_id_)
    _send_more_questions(sender, phone_id_, lang)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def handle_general_followup(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    is_greeting = _contains_signal(prompt_lower, GREETING_WORDS)
    is_reset = _contains_signal(prompt_lower, RESET_KEYWORDS)

    if is_greeting or is_reset:
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_)
        save_single_user_state(sender)
        return

    if _contains_signal(prompt_lower, YES_RESPONSES):
        ask_map = {
            "shona": "Bvunzai mubvunzo wenyu.", "ndebele": "Ngiyacela ubuze umbuzo wakho.",
            "tonga": "Amubuye mubuyo", "chinyanja": "Chonde funsani funso lanu.",
            "bemba": "Nomba, ipusha ilipusho lyobe.", "lozi": "Nkumbira ubuze mubvuzo wako.",
        }
        send(ask_map.get(lang, "Please ask your question."), sender, phone_id_)
        state["step"] = "general_question"
        save_single_user_state(sender)
        return

    if _contains_signal(prompt_lower, NO_RESPONSES):
        _ask_purchase_interest(sender, phone_id_, lang)
        return

    if len(prompt_lower.split()) > 2:
        _send_thinking(sender, phone_id_, lang)

    reply = ask_gemini_general(prompt, lang, sender=sender)
    send(reply, sender, phone_id_)
    _send_more_questions(sender, phone_id_, lang)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def ask_follow_up_question(sender, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    followup_map = {
        "shona": "Pane chimwe chandingakubatsira nacho here?", "ndebele": "Ingabe kukhona okunye engingakusiza ngakho?",
        "tonga": "Hena muli amubuyo umbi?", "chinyanja": "Kodi pali zina zomwe ndingakuthandizireni?",
        "bemba": "Kuli fintu fimbi ifyo ningamwafwilisha?", "lozi": "Ki sina sika ni ka thusa ka sona",
    }
    send(followup_map.get(lang, "Is there anything else I can help you with?"), sender, phone_id_)
    state["step"] = "follow_up"
    save_single_user_state(sender)


def handle_main_menu(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    is_reset = _contains_signal(prompt_lower, RESET_KEYWORDS)
    is_greeting = _contains_signal(prompt_lower, GREETING_WORDS)

    if is_greeting or is_reset:
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_)
        save_single_user_state(sender)
        return

    current_step = state.get("step")

    step_router = {
        "ask_another_week": handle_another_week,
        "cervical_more_info": handle_cervical_more_info,
        "cervical_question_number": handle_cervical_question_number,
        "keep_learning": handle_keep_learning,
        "follow_up": handle_follow_up,
        "product_inquiry": handle_purchase_response,
        "confirm_purchase": handle_purchase_confirmation,
        "general_followup": handle_general_followup,
    }
    if current_step in step_router:
        step_router[current_step](sender, prompt, phone_id_)
        return

    if state.get("step") == "choose_info_type":
        _handle_choose_info_type(sender, prompt, phone_id_)
        return

    if state.get("step") == "ask_week":
        _handle_ask_week_step(sender, prompt, phone_id_)
        return

    if state.get("step") == "maternal_question_choice":
        _handle_maternal_question_choice(sender, prompt, phone_id_)
        return

    if state.get("step") == "cervical_question_choice":
        _handle_cervical_question_choice(sender, prompt, phone_id_)
        return

    maternal_keywords = ["pamuviri", "pakati", "pregnancy", "pregnant", "baby", "maternal", "nhumbu"]
    question_words = ["what", "how", "when", "why", "can", "should", "kuti", "sei", "ngani", "kodi", "bwanji",
                       "chifukwa", "ndeipi", "ndiani", "nzira", "zviratidzo", "zizindikiro"]
    is_direct_question = (
        any(k in prompt_lower for k in maternal_keywords) and any(q in prompt_lower for q in question_words)
    )

    if is_direct_question:
        _send_thinking(sender, phone_id_, lang)
        gemini_response = ask_gemini(prompt, lang, sender=sender)
        send(gemini_response, sender, phone_id_)
        ask_follow_up_question(sender, phone_id_)
        save_single_user_state(sender)
        return

    gemini_reply = ask_gemini_general(prompt, lang, sender=sender)
    send(gemini_reply, sender, phone_id_)
    _send_more_questions(sender, phone_id_, lang)
    state["step"] = "general_followup"
    save_single_user_state(sender)


def _handle_choose_info_type(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if prompt_lower in ["1", "general", "information", "info", "ruzivo", "ulwazi", "zambiri"]:
        if state.get("topic") == "maternal":
            state["step"] = "ask_week"
            week_map = {
                "shona": "Ndapota isa vhiki re pamuviri ", "ndebele": "Sicela ufake iviki lokukhulelwa ",
                "chinyanja": "Chonde lowetsani sabata la pakati ", "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
                "tonga": " Amubike namba yama wiki yo mwaba andaa (yo mwaba akaati)",
                "bemba": "Napapita, ingisha umulungu wa pa nkundi ",
            }
            send(week_map.get(lang, "Please enter your pregnancy week number:"), sender, phone_id_)
        elif state.get("topic") == "cervical":
            cervical_data = get_cervical_data(lang)
            if cervical_data and len(cervical_data) > 0:
                send(str(cervical_data[0]), sender, phone_id_)
            else:
                no_data_map = {
                    "shona": "Ndine urombo, handina kuwana ruzivo rwe cervical cancer parizvino.",
                    "ndebele": "Uxolo, anginayo imininingwane ye-cervical cancer okwamanje.",
                    "chinyanja": "Pepani, sindinapeze zambiri za cervical cancer panopa.",
                    "lozi": "Ndine u luvile, sina kungafumula zintu za kankere ya sibete sunu.",
                }
                send(no_data_map.get(lang, "Sorry, I couldn't find cervical cancer information at the moment."), sender, phone_id_)
            ask_cervical_more_info(sender, phone_id_)
        save_single_user_state(sender)
        return

    elif prompt_lower in ["2", "specific", "question", "questions", "mubvunzo", "umbuzo", "funso"]:
        if state.get("topic") == "maternal":
            state["step"] = "maternal_question_choice"
            q_map = {
                "shona": "Sarudza mubvunzo:\n1. Ndezvikaita zviratidzo zvepamuviri?\n2. Ndeapi marairiro ezvokudya?\n3. Ndingafanire kuona chiremba riini?",
                "ndebele": "Khetha umbuzo:\n1. Ngabe yiziphi izimpawu zesisu?\n2. Ngabe yimaphi amathiphu okudla?\n3. Ngabe kufanele ngibone udokotela nini?",
                "chinyanja": "Sankhani funso:\n1. Ndi zizindikiro zotani za pakati?\n2. Ndi malangizo otani okudya?\n3. Ndingafunire kuona dokotala liti?",
                "lozi": "U ka khetha mubuzo noma u buze mubuzo wa wena.\n1. Zibonelelo ze ku imelela mwana zezi ntini?\n2. Ni maano a ku nwa zintu za bupilo a ka landelwa?\n3. Nini nka ya kwa dokotela?",
                "tonga": "Sala mubuzyo:\n1. Hena nga mwaiziba buti kutu muntu uli andaa olo uli akaati?\n2. Hena zilyo nzi zyo elede kulya mukaintu uli andaa?\n3. Hena chiindi nzi cho diyelede kubona ba dokata?",
                "bemba": "Sala ilipusho:\n1. Finshi nigeshibilako ukutila ndi pabukulu?\n2. Mabumba ya fyakulya nshi fwile ukulya?\n3. Nfwile ukumona dokota lisa?",
            }
            send(q_map.get(lang, "You can choose a question or ask any of your own.\n1. What are common pregnancy symptoms?\n2. What nutrition tips should I follow?\n3. When should I see a doctor?"), sender, phone_id_)
        elif state.get("topic") == "cervical":
            state["step"] = "cervical_question_choice"
            cq_map = {
                "shona": "Sarudza mubvunzo:\n1. Chii chinonzi cervical cancer?\n2. Ndezvipi zviratidzo zvekutanga zvecervical cancer?\n3. Chii chinokonzera cervical cancer?",
                "ndebele": "Khetha umbuzo:\n1. Yini i-cervical cancer?\n2. Ngabe yiziphi izimpawu zokuqala ze-cervical cancer?\n3. Yini ebangela i-cervical cancer?",
                "chinyanja": "Sankhani funso:\n1. Ndi chiyani cervical cancer?\n2. Ndi zizindikiro zotani zoyamba za cervical cancer?\n3. Ndi chiyani chimayambitsa cervical cancer?",
                "lozi": "U ka khetha mubuzo noma u buze mubuzo wa wena.\n1. Kankere ya sibete sa bomme ki yini?\n2. Zibonelelo za kutanga za kankere ya sibete zezi ntini?\n3. Zini zi bakela kankere ya sibete?",
                "tonga": "Sarudza mubvuzo:\n1. Kansa ya mulomo wa cibeleko nzi?\n2. Zizyo zyakutanga zya kansa ya mulomo wa cibeleko nzi?\n3. Chiyambitsa kansa ya mulomo wa cibeleko nzi?",
                "bemba": "Sala ilipusho:\n1. Bushe Cervical cancer nichinshi?\n2. Finshi ningamwenako ukutila ninkwata cervical cancer?\n3. Finshi ifileta Cervical cancer?",
            }
            send(cq_map.get(lang, "You can choose a question or ask any of your own.\n1. What is cervical cancer?\n2. What are the early symptoms of cervical cancer?\n3. What causes cervical cancer?"), sender, phone_id_)
        save_single_user_state(sender)
        return
    else:
        invalid_map = {
            "shona": "Pindura ne '1' kuti uwane ruzivo kana '2' kuti ubvunze mibvunzo.",
            "ndebele": "Phendula ngo-'1' ukuze uthole ulwazi noma '2' ukuze ubuze imibuzo.",
            "chinyanja": "Yankhani ndi '1' kuti mupeze zambiri kapena '2' kuti mufunse mafunso.",
            "tonga": "Ingula a '1' kuti uzibe zinji'2' kuti ubuzye", "bemba": "Yasuka na '1' ukuti usanga ifingi '2' Ukuti wipushe ilipusho.",
            "lozi": "Arabela ka '1' ku fumana litaba kamba '2' ku buza lipuzo.",
        }
        send(invalid_map.get(lang, "Please reply '1' for information or '2' for questions."), sender, phone_id_)


def _handle_ask_week_step(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    try:
        week = int(re.sub(r"\D", "", prompt_lower))
        if 1 <= week <= 40:
            info_text = get_pregnancy_data(lang)
            week_labels = {
                "shona": "Vhiki", "ndebele": "Iviki", "chinyanja": "Sabata", "lozi": "Sunda",
                "bemba": "Umulungu", "tonga": "Nhwiiiki",
            }
            label = week_labels.get(lang, "Week")
            pattern = rf"\*{label} {week}:.*?(?=\*{label} {week+1}:|\Z)"
            match = re.search(pattern, info_text, re.S)
            if match:
                header_map = {
                    "shona": f"Ruzivo rwe *Vhiki {week}:*\n\n", "ndebele": f"Ulwazi lwe *Iviki {week}:*\n\n",
                    "chinyanja": f"Zambiri za *Sabata {week}:*\n\n", "lozi": f"Yezi zintu za lwisisa ka bonya ku *Sunda {week}:*\n\n",
                    "bemba": f"Icibeela ca *Mulungu {week}:*\n\n", "tonga": f"Zinji zya *Wiiki {week}:*\n\n",
                }
                header = header_map.get(lang, f"Here's information for *Week {week}:*\n\n")
                send(f"{header}{match.group(0)}", sender, phone_id_)
                ask_another_week(sender, phone_id_)
            else:
                no_week_map = {
                    "shona": "Hapana ruzivo rwevhiki iyi.", "ndebele": "Alukho ulwazi lwaleviki.",
                    "chinyanja": "Palibe zambiri za sabata ili.", "lozi": "Sina zintu za ku fumwa ka viki ye.",
                    "bemba": "Tapali ifilifyonse pa mulungu uyu.", "tonga": "Kunyina zinji zya wiiki iyi.",
                }
                send(no_week_map.get(lang, "No data available for that week."), sender, phone_id_)
                ask_another_week(sender, phone_id_)
        else:
            invalid_week_map = {
                "shona": "Ndapota pinda nhamba chaiyo yevhiki kubva pa 1 kusvika pa 40.",
                "ndebele": "Sicela ufake inombolo yeviki evumelekile ephakathi kuka-1 no-40.",
                "chinyanja": "Chonde lowetsani nambala yoyenera ya sabata kuchokera pa 1 mpaka 40.",
                "lozi": "Ndapota faka linomolo la viki le li le ka 1 ku ya ka 40.",
                "bemba": "Chisuma, ingisha umulungu ukufuma pa 1 ukufika pa 40.",
                "tonga": "Ndakomba, Njisya namba ya week kutalikila a 1 kusika a 40.",
            }
            send(invalid_week_map.get(lang, "Please enter a valid week number between 1 and 40."), sender, phone_id_)
            ask_another_week(sender, phone_id_)
    except ValueError:
        invalid_week_map = {
            "shona": "Ndapota pinda nhamba chaiyo yevhiki kubva pa 1 kusvika pa 40.",
            "ndebele": "Sicela ufake inombolo yeviki evumelekile ephakathi kuka-1 no-40.",
        }
        send(invalid_week_map.get(lang, "Please enter a valid week number between 1 and 40."), sender, phone_id_)
        ask_another_week(sender, phone_id_)


def _handle_maternal_question_choice(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if prompt_lower in ["1", "symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
        sym_map = {
            "shona": "Zviratidzo zvepamuviri zvinosanganisira kusvotwa, kuneta, kuvava mazamu, uye kuchinja mweya.",
            "ndebele": "Izimpawu zesisu zihlanganisa isicanucanu, ukukhathala, ubuhlungu bezebelé, nokushintsha kwemizwa.",
            "chinyanja": "Zizindikiro za pakati zimaphatikizapo kusanza, kulemba, kubvutika mabele, ndi kusintha kwa maganizo.",
            "lozi": "Limpande ze twayelehileng za buimana li akaretsa ho nyekeloa ke pelo, kukhathala, kubaba kwa matete ni kupotoloka kwa maikuto.",
            "tonga": "Zitondezyo zyakuba ada zyboneka obu: Kuseluka kumoyo, Nkolo kucisa, kukola, kukatala kapati, a kuchinja chinja kwamizezo.",
            "bemba": "Ifyo wingeshibilako ukuti naukwata ifumo, Umuselu, ukunakilila kwama bele, no kuchinja kwamisango ne fichitwa.",
        }
        send(sym_map.get(lang, "Common pregnancy symptoms include nausea, fatigue, breast tenderness, and mood swings."), sender, phone_id_)
    elif prompt_lower in ["2", "nutrition", "zvokudya", "ukudla", "kudya", "kulya", "ukulya"]:
        nut_map = {
            "shona": "Marairiro ezvokudya: Idya chikafu chakaringana, wedzera folic acid uye iron, uye nwa mvura yakawanda.",
            "ndebele": "Amathiphu okudla: Yidla ukudla okunempilo, khulisa i-folic acid ne-iron, futhi uhlale unamandla.",
            "chinyanja": "Malangizo okudya: Idyani chakudya chabwino, onjezerani folic acid ndi iron, ndipo muzikhala ndi madzi.",
            "lozi": "Litaba za swakudya: Ja swakudya se se lekalekanang, engetsa kufumana folic acid ni iron, mi u nne u nwa mezi a mangi.",
            "tonga": "Zilyo zyo mwelede kulya: Amulye zilyo zyelede, engesha folic acid ni iron, anilizyo nwa maanzi amanji.",
            "bemba": "Amabumba ya fyakulya: Lya ifya kulya ifya balansa, Mulungizye zilyo zigisi folic acid a iron inji, Amunye menda manji.",
        }
        send(nut_map.get(lang, "Nutrition tips: Eat balanced meals, increase folic acid and iron intake, and stay hydrated."), sender, phone_id_)
    elif prompt_lower in ["3", "doctor", "chiremba", "udokotela", "dokotala", "dokota"]:
        doc_map = {
            "shona": "Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.",
            "ndebele": "Iya kudokotela uma unobuhlungu obukhulu, ukuphuma kwegazi okukhulu, noma imfiva ephezulu.",
            "chinyanja": "Pitani kudokotala ngati muli ndi kupweteka kwakukulu, kutuluka magazi ambiri, kapena malungo apamwamba.",
            "lozi": "Bona ngaka kapili ha u ka ba ni buhlungu bo boholo, kuelwa mali a mangi, kamba mufufutso o mutuna.",
            "tonga": "Bona dokotela cakufwambana, kutuluka magazi amanji, naa malungo apamwamba.",
            "bemba": "Kamubona musilisi mukufwambaana kuti mwamvwa kucisa kapati, kuswa bulowa bunji, naa kupya kapati mubili.",
        }
        send(doc_map.get(lang, "See a doctor immediately if you experience severe pain, heavy bleeding, or high fever."), sender, phone_id_)
    else:
        _send_thinking(sender, phone_id_, lang)
        send(ask_gemini(prompt, lang, sender=sender), sender, phone_id_)

    ask_follow_up_question(sender, phone_id_)
    save_single_user_state(sender)


def _handle_cervical_question_choice(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if prompt_lower in ["1", "what is it", "what is cervical cancer", "chii", "yini", "chiyani"]:
        cc_what_map = {
            "shona": "Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.",
            "ndebele": "I-cervical cancer yisifo se-cervix, ingxenye engezansi yesibeletho ehlobene nesibeletho. Yisifo somhlaza sesibili esivame kakhulu emhlabeni wonke futhi yisifo esivame kakhulu kwabesifazane eZambia.",
            "chinyanja": "Cervical cancer ndi matenda a cervix, gawo lotsika la chibereko lomwe limagwirizana ndi chibereko.",
        }
        send(cc_what_map.get(lang, "Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. It is the second most common female malignancy worldwide and the most common in females in Zambia. It is a preventable and treatable disease, especially when detected early."), sender, phone_id_)
    elif prompt_lower in ["2", "symptoms", "early symptoms", "zviratidzo", "izimpawu", "zizindikiro", "zitondezyo"]:
        send("In its early stages, cervical cancer often has no noticeable symptoms. This is why regular screening is so important. As the cancer progresses, symptoms may include unusual vaginal bleeding, foul-smelling vaginal discharge, or pain during sexual intercourse.", sender, phone_id_)
    elif prompt_lower in ["3", "causes", "what causes it", "chikonzero", "izimbangela", "zoyambitsa"]:
        send("In almost all cases, cervical cancer is caused by persistent infection with the Human Papilloma Virus (HPV), a very common sexually transmitted virus. While the body clears the virus in most people, a persistent infection can lead to abnormal cell changes that may eventually develop into cancer.", sender, phone_id_)
    else:
        _send_thinking(sender, phone_id_, lang)
        send(ask_gemini_cancer(prompt, lang, sender=sender), sender, phone_id_)

    ask_follow_up_question(sender, phone_id_)
    save_single_user_state(sender)


def _ask_purchase_interest(sender, phone_id_, lang):
    state = user_states[sender]
    ask_map = {
        "shona": "Ungada here kutenga zvimwe zvezvigadzirwa zvedu? ", "ndebele": "Ungathanda ukuthenga noma yimuphi imikhiqizo yethu? ",
        "chinyanja": "Kodi mukufuna kugula zinthu zina mu zithu zathu? ", "tonga": "Mulakonzya kuyanda kuula zimwi zintu zyesu? ",
        "bemba": "Bushe kuti mwatemwa ukushita ifipe fyesu fimo? ", "lozi": "Kana u bata ku landa swakupila sa luna? ",
    }
    send(ask_map.get(lang, "Would you like to purchase any of our products?"), sender, phone_id_)
    state["step"] = "shop_interest"
    save_single_user_state(sender)


def _send_shop_categories(sender, phone_id_, lang):
    lines = []
    header_map = {
        "shona": "🛒 Makategi eZvigadzirwa:\n", "ndebele": "🛒 Imigqa Yemikhiqizo:\n",
        "chinyanja": "🛒 Mitundu ya Zinthu:\n", "tonga": "🛒 Misela ya Zintu:\n",
        "bemba": "🛒 Imisango ya fipe:\n", "lozi": "🛒 Mibeko ya Swakupila:\n",
    }
    lines.append(header_map.get(lang, "🛒 Product Categories:\n"))

    for idx, (cat_name, items) in enumerate(products_by_category.items(), 1):
        lines.append(f"*{idx}. {cat_name}*")
        for item in items[:2]:
            lines.append(f"   • {item['name']} — {item['price']}")
        if len(items) > 2:
            lines.append(f"   ...and {len(items) - 2} more")
        lines.append("")

    prompt_map = {
        "shona": "Tumira nhamba yekategi kuti uone zvigadzirwa zvose, kana udza zita rechigadzirwa chaunoda kutenga.",
    }
    lines.append(prompt_map.get(lang, "Send the category number to see all products, or tell us the name of the product you'd like to order."))

    send("\n".join(lines), sender, phone_id_)
    state = user_states[sender]
    state["step"] = "shop_browse"
    save_single_user_state(sender)


def _interpret_shop_intent(prompt_lower):
    browse_signals = [
        "yes", "yeah", "yep", "please", "sure", "ok", "okay", "alright", "ehe", "hongu", "ndizvo", "inde", "yebo",
        "product", "products", "what do you have", "what have you got", "show me", "available", "categories",
        "add more", "something else", "buy", "purchase", "looking for",
        "zvigadzirwa", "zvinhu", "imikhiqizo", "zinthu", "imisansa", "swakupila",
    ]
    decline_signals = [
        "no", "nah", "nope", "not really", "not now", "that's all", "that is all", "done", "finish", "complete",
        "checkout", "later", "goodbye", "bye", "hapana", "kwete", "aiwa", "a'a", "ayi", "cha",
    ]
    if _contains_signal(prompt_lower, browse_signals):
        return "browse"
    if _contains_signal(prompt_lower, decline_signals):
        return "decline"
    return "unknown"


def handle_shop_interest(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id_, lang)
    elif intent == "decline":
        send("Alright! Have a nice day. Say 'hi' if you have more questions.", sender, phone_id_)
        reset_conversation(sender)
    else:
        _send_shop_categories(sender, phone_id_, lang)


def handle_shop_browse(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    categories = list(products_by_category.keys())

    if prompt_lower.isdigit():
        idx = int(prompt_lower) - 1
        if 0 <= idx < len(categories):
            cat_name = categories[idx]
            all_items = products_by_category[cat_name]
            lines = [f"🏥 *{cat_name}*\n"]
            for i, item in enumerate(all_items, 1):
                lines.append(f"{i}. {item['name']}")
                lines.append(f"   💰 {item['price']} | 📦 {item['availability']}")
                lines.append(f"   {item['description']}\n")
            lines.append("\nWould you like to order any of these products? Reply 'yes' and tell us the product name, or 'no'.")
            send("\n".join(lines), sender, phone_id_)
            state["step"] = "shop_order_decision"
            state["shop_category"] = cat_name
            save_single_user_state(sender)
            return
        else:
            send(f"Invalid number. Please choose between 1 and {len(categories)}.", sender, phone_id_)
            return

    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id_, lang, matched["name"])
        return

    _send_shop_categories(sender, phone_id_, lang)


def handle_shop_order_decision(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, YES_RESPONSES):
        send("Great! Please type the name of the product you'd like to order.", sender, phone_id_)
        state["step"] = "shop_product_name"
        save_single_user_state(sender)
    elif _contains_signal(prompt_lower, NO_RESPONSES):
        send("Alright! Would you like to see other categories?", sender, phone_id_)
        state["step"] = "shop_more_categories"
        save_single_user_state(sender)
    else:
        all_products_flat = [p for items in products_by_category.values() for p in items]
        matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            _ask_quantity(sender, phone_id_, lang, matched["name"])
        else:
            send("Please reply 'yes' or 'no'.", sender, phone_id_)


def handle_shop_product_name(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id_, lang, matched["name"])
    else:
        send("I couldn't find that product. Please type the exact product name from the list.", sender, phone_id_)


def handle_shop_more_categories(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id_, lang)
    elif intent == "decline":
        send("Alright! Have a nice day. Say 'hi' if you have more questions.", sender, phone_id_)
        reset_conversation(sender)
    else:
        _send_shop_categories(sender, phone_id_, lang)


def _ask_quantity(sender, phone_id_, lang, product_name):
    state = user_states[sender]
    send(f"Great! How many *{product_name}* would you like?", sender, phone_id_)
    state["step"] = "shop_quantity"
    save_single_user_state(sender)


def handle_shop_quantity(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    qty_match = re.search(r"\d+", prompt.strip())
    if qty_match:
        qty = int(qty_match.group())
        cart = state.setdefault("cart", [])
        cart.append({
            "product": state.get("shop_selected_product", "Unknown"),
            "price": state.get("shop_selected_price", "N/A"),
            "quantity": qty,
        })
        send(f"✅ *{state.get('shop_selected_product')}* x{qty} added! Would you like to add anything else?", sender, phone_id_)
        state["step"] = "shop_add_more"
        save_single_user_state(sender)
    else:
        send("Please enter a number (e.g. 1, 2, 3).", sender, phone_id_)


def _save_orders_to_redis(sender, cart, address):
    if not redis_client:
        return
    user_id = user_states[sender].get("user_id", sender)
    for item in cart:
        order = {
            "user_id": user_id, "sender": sender, "product": item["product"], "price": item["price"],
            "quantity": item["quantity"], "address": address, "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }
        try:
            order_key = f"orders:{sender}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            redis_client.set(order_key, json.dumps(order))
        except Exception as e:
            logging.error(f"Error saving order: {e}")


def _send_order_confirmation(sender, phone_id_, lang, cart, address):
    parts = ["✅ *Order Confirmed!*", ""]
    for item in cart:
        parts.append(f"  📦 {item['product']} x{item['quantity']} — {item['price']}")
    parts.append("")
    parts.append(f"📍 Delivery address: {address}")
    parts.append("We'll be in touch shortly. Thank you! 😊")
    send("\n".join(parts), sender, phone_id_)


def handle_shop_address(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    address = prompt.strip()
    cart = state.get("cart", [])
    _save_orders_to_redis(sender, cart, address)
    _send_order_confirmation(sender, phone_id_, lang, cart, address)
    reset_conversation(sender)


def handle_shop_add_more(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id_, lang, matched["name"])
        return

    intent = _interpret_shop_intent(prompt_lower)
    if intent == "browse":
        _send_shop_categories(sender, phone_id_, lang)
    else:
        send("Great! Please provide your delivery address (town, area, and any helpful details).", sender, phone_id_)
        state["step"] = "shop_address"
        save_single_user_state(sender)


def extract_products_by_category(category_name):
    try:
        return products_by_category.get(category_name, [])
    except Exception as e:
        logging.error(f"Error extracting products for category {category_name}: {e}")
        return []


def format_products_for_display(products_list, lang):
    if not products_list:
        return "No products currently available."
    products_text = "🏥 Health Products:\n\n"
    for i, product in enumerate(products_list, 1):
        name = product.get("name", "Unknown Product")
        price = product.get("price", "Price not available")
        availability = product.get("availability", "Availability not specified")
        products_text += f"{i}. {name}\n   💰 Price: {price}\n   📦 Availability: {availability}\n\n"
    products_text += "Select a product by telling us the number."
    return products_text


def handle_purchase_response(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, NO_RESPONSES):
        send("Thank you! Have a nice day. If you have more questions, start over by saying 'hi'.", sender, phone_id_)
        reset_conversation(sender)
        return
    elif _contains_signal(prompt_lower, YES_RESPONSES):
        topic = state.get("topic")
        category_map = {"maternal": "Maternal Health", "cervical": "Cervical Cancer"}
        category_name = category_map.get(topic, "General")
        products = extract_products_by_category(category_name)
        if products:
            send(format_products_for_display(products, lang), sender, phone_id_)
        else:
            send("Sorry, no products are currently available in this category. We recommend visiting our clinic for more information.", sender, phone_id_)
        send("Would you like to proceed with purchasing any of these products?", sender, phone_id_)
        state["step"] = "confirm_purchase"
        save_single_user_state(sender)
    else:
        send("I didn't understand. Please reply: Would you like to purchase products?", sender, phone_id_)


def handle_purchase_confirmation(sender, prompt, phone_id_):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, NO_RESPONSES):
        send("Alright. Thank you! If you have more questions, start over by saying 'hi'.", sender, phone_id_)
        reset_conversation(sender)
    elif _contains_signal(prompt_lower, YES_RESPONSES):
        send("Thank you! We'll contact you shortly for more details about your purchase.", sender, phone_id_)
        reset_conversation(sender)
    else:
        send("I didn't understand. Please reply: Would you like to proceed with purchasing?", sender, phone_id_)


# ─────────────────────────────────────────────
#  Gemini helper functions (stateless)
# ─────────────────────────────────────────────

def _get_lang_enforce(lang: str) -> str:
    return {
        "shona": "Pindura muchiShona chete. Usashandise Chirungu.",
        "ndebele": "Phendula ngesiNdebele kuphela. Ungasebenzisi isiNgisi.",
        "chinyanja": "Yankhani mu Chichewa/Chinyanja basi. Osagwiritsa ntchito Chingerezi.",
        "lozi": "Arabela ka Silozi feela. U se ke wa sebelisa Siingelesi.",
        "bemba": "Yasuka mu Cibemba fye. Wibonfya icingeleshi.",
        "tonga": "Mupandule mu Chitonga buyo. Mutabelezyi Ciingelezi.",
    }.get(lang, "Respond in English only.")


def _get_fallback(lang: str) -> str:
    return {
        "shona": "Pane dambudziko pakupindura mubvunzo wako.",
        "ndebele": "Kunenkinga ekuphenduleni umbuzo wakho.",
        "chinyanja": "Pali vuto popanga yankho la funso lanu.",
        "tonga": "Kuli ipenzi mukupandula mubuzyo wanu.",
        "bemba": "Cabulanda, kuliko ubwafya pakwasuka kuyasuka ilipusho lyobe.",
        "lozi": "Ku na bothata ka ku arabela lipuzo la hao.",
    }.get(lang, "Sorry, there was a problem getting an answer.")


def build_conversation_context(sender: str, max_turns: int = 6) -> str:
    history = get_user_conversation(sender)
    if not history:
        return ""
    recent = history[-(max_turns * 2 + 1):-1]
    lines = []
    for entry in recent:
        role = entry.get("role", "")
        message = entry.get("message", "").strip()
        if not message:
            continue
        if role == "bot" and len(message) < 20:
            continue
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {message}")
    if not lines:
        return ""
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def _call_gemini(prompt: str, fallback: str) -> str:
    if not gen_api:
        return fallback
    try:
        gemini_model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
        response = gemini_model.generate_content(prompt)
        text = response.text
        if text and text.strip():
            return text.strip()
    except Exception as e:
        logging.error(f"[_call_gemini Error] {type(e).__name__}: {e}")
    return fallback


def ask_gemini(question: str, lang: str = "english", sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)
    context = build_conversation_context(sender) if sender else ""
    instruction_body = (
        "You are a maternal health assistant. Answer the following question clearly, "
        "simply, and with accurate health information:\n\n"
    )
    prompt = f"{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"
    return _call_gemini(prompt, fallback)


def ask_gemini_cancer(question: str, lang: str = "english", sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)
    context = build_conversation_context(sender) if sender else ""
    instruction_body = (
        "You are a cervical cancer health assistant. Answer the following question "
        "clearly and simply:\n\n"
    )
    prompt = f"{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"
    return _call_gemini(prompt, fallback)


def ask_gemini_general(question: str, lang: str, sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)
    context = build_conversation_context(sender) if sender else ""

    company_address = "No. 50 Lunsemfwa Rd, Kalundu, Lusaka, Zambia"
    company_email = "hello@dawa-health.com"
    company_website = "https://dawa-health.com/"
    company_phone = "+260 571 376 677"

    instruction_body = (
        "You are a professional health assistant specializing in maternal health, sexual "
        "reproductive health and cervical cancer for Dawa Health. Answer the user question "
        "using correct and evidence-based health information. DO NOT start with phrases like "
        "Okay, Sure, or Let me explain. Start directly with the answer. Include a brief "
        "disclaimer at the end stating that this information does not replace a doctor "
        "evaluation. If asked about home visits, Dawa Health clinicians do home visits. "
        f"Contact: email={company_email}, phone={company_phone}, address={company_address}, "
        f"website={company_website}.\n\n"
    )
    prompt = f"{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"
    return _call_gemini(prompt, fallback)


def ask_cervical_more_info(sender, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    send("Would you like to get more information about cervical cancer?", sender, phone_id_)
    state["step"] = "cervical_more_info"
    save_single_user_state(sender)


def ask_cervical_question_number(sender, phone_id_):
    state = user_states[sender]
    send("Enter a question number from 1 to 100:", sender, phone_id_)
    state["step"] = "cervical_question_number"
    save_single_user_state(sender)


def ask_keep_learning(sender, phone_id_):
    state = user_states[sender]
    send("Would you like to keep learning more about cervical cancer?", sender, phone_id_)
    state["step"] = "keep_learning"
    save_single_user_state(sender)


def handle_cervical_more_info(sender, prompt, phone_id_):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()
    if _contains_signal(prompt_lower, YES_RESPONSES):
        ask_cervical_question_number(sender, phone_id_)
    elif _contains_signal(prompt_lower, NO_RESPONSES):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id_)
    else:
        send("I didn't understand. Would you like to get more information?", sender, phone_id_)


def handle_cervical_question_number(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    try:
        question_num = int(re.sub(r"\D", "", prompt))
        if 1 <= question_num <= 100:
            data_tuple = get_cervical_data(lang)
            question_found = False
            for i, item in enumerate(data_tuple):
                if f"*Question {question_num}:" in str(item):
                    question_content = str(item)
                    if i + 1 < len(data_tuple) and "Answer" in str(data_tuple[i + 1]):
                        question_content += "\n" + str(data_tuple[i + 1])
                    send(question_content, sender, phone_id_)
                    question_found = True
                    ask_keep_learning(sender, phone_id_)
                    break
            if not question_found:
                send(f"Sorry, I couldn't find question number {question_num}. Please try another number from 1 to 100.", sender, phone_id_)
                ask_cervical_question_number(sender, phone_id_)
        else:
            send("Please enter a number between 1 and 100 only.", sender, phone_id_)
            ask_cervical_question_number(sender, phone_id_)
    except ValueError:
        send("Please enter a valid number between 1 and 100.", sender, phone_id_)
        ask_cervical_question_number(sender, phone_id_)


def handle_keep_learning(sender, prompt, phone_id_):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()
    if _contains_signal(prompt_lower, YES_RESPONSES):
        ask_cervical_question_number(sender, phone_id_)
    elif _contains_signal(prompt_lower, NO_RESPONSES):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id_)
    else:
        send("I didn't understand. Would you like to keep learning?", sender, phone_id_)


def ask_another_week(sender, phone_id_):
    state = user_states[sender]
    send("Would you like to learn about other pregnancy weeks?", sender, phone_id_)
    state["step"] = "ask_another_week"
    save_single_user_state(sender)


def handle_another_week(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, YES_RESPONSES):
        state["step"] = "ask_week"
        week_map = {
            "shona": "Ndapota isa vhiki re pamuviri ", "ndebele": "Sicela ufake iviki lokukhulelwa ",
            "chinyanja": "Chonde lowetsani sabata la pakati ", "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
            "tonga": "Mulembesye nambala ya week ya bubulemi yenu", "bemba": "Chisuma, ingisheni nambala yamilungu mwaba pabukulu ",
        }
        send(week_map.get(lang, "Please enter your pregnancy week number "), sender, phone_id_)
        save_single_user_state(sender)
    elif _contains_signal(prompt_lower, NO_RESPONSES):
        state["step"] = "product_inquiry"
        state["topic"] = "maternal"
        send("Thank you! Would you like to purchase maternal health products? We offer:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits", sender, phone_id_)
        save_single_user_state(sender)
    else:
        send("I didn't understand. Would you like to learn about other weeks?", sender, phone_id_)


# ─────────────────────────────────────────────
#  MAIN CONVERSATION ROUTER
# ─────────────────────────────────────────────

def handle_conversation_state(sender, prompt, phone_id_):
    state = user_states[sender]
    current_step = state.get("step")

    if current_step not in ["language_detection", "registration"]:
        maybe_update_language(sender, prompt)
        state = user_states[sender]

    if current_step == "human_agent_chat":
        relayed = relay_user_message_to_agent(sender, prompt, phone_id_)
        if not relayed:
            user_states[sender]["step"] = "main_menu"
            user_states[sender].pop("agent_phone", None)
            save_single_user_state(sender)
            send("Your agent session has ended. Returning you to Rudo. How can I help?", sender, phone_id_)
        return

    if current_step == "waiting_for_agent":
        if prompt.strip().upper() == "CANCEL":
            if redis_client:
                try:
                    redis_client.delete(_agent_request_key(sender))
                    redis_client.delete(_agent_rejections_key(sender))
                except Exception:
                    pass
            user_states[sender]["step"] = "main_menu"
            save_single_user_state(sender)
            send("Your request has been cancelled. Returning you to Rudo.", sender, phone_id_)
        else:
            check_agent_request_timeout(sender, phone_id_)
        return

    if current_step not in ["language_detection", "registration"]:
        if is_human_agent_request(prompt):
            send("🔍 Looking for a human agent to assist you...\nPlease wait. An agent will accept or decline within about a minute.\nType *CANCEL* if you'd like to return to Rudo.", sender, phone_id_)
            user_states[sender]["step"] = "waiting_for_agent"
            save_single_user_state(sender)
            notify_agents_of_request(sender, phone_id_)
            return

    prompt_lower = prompt.lower().strip()
    is_greeting = _contains_signal(prompt_lower, GREETING_WORDS)
    is_reset = _contains_signal(prompt_lower, RESET_KEYWORDS)

    if (is_greeting or is_reset) and current_step not in ["language_detection", "registration"]:
        reset_conversation(sender)
        lang = user_states[sender]["language"]
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_)
        return

    current_step = state.get("step")

    routes = {
        "language_detection": lambda: handle_language_detection(sender, prompt, phone_id_) if state.get("first_message", True) else handle_main_menu(sender, prompt, phone_id_),
        "registration": lambda: handle_registration(sender, prompt, phone_id_),
        "ask_another_week": lambda: handle_another_week(sender, prompt, phone_id_),
        "cervical_more_info": lambda: handle_cervical_more_info(sender, prompt, phone_id_),
        "cervical_question_number": lambda: handle_cervical_question_number(sender, prompt, phone_id_),
        "keep_learning": lambda: handle_keep_learning(sender, prompt, phone_id_),
        "follow_up": lambda: handle_follow_up(sender, prompt, phone_id_),
        "product_inquiry": lambda: handle_purchase_response(sender, prompt, phone_id_),
        "confirm_purchase": lambda: handle_purchase_confirmation(sender, prompt, phone_id_),
        "shop_interest": lambda: handle_shop_interest(sender, prompt, phone_id_),
        "shop_browse": lambda: handle_shop_browse(sender, prompt, phone_id_),
        "shop_order_decision": lambda: handle_shop_order_decision(sender, prompt, phone_id_),
        "shop_product_name": lambda: handle_shop_product_name(sender, prompt, phone_id_),
        "shop_more_categories": lambda: handle_shop_more_categories(sender, prompt, phone_id_),
        "shop_add_more": lambda: handle_shop_add_more(sender, prompt, phone_id_),
        "shop_quantity": lambda: handle_shop_quantity(sender, prompt, phone_id_),
        "shop_address": lambda: handle_shop_address(sender, prompt, phone_id_),
        "general_followup": lambda: handle_general_followup(sender, prompt, phone_id_),
        "general_question": lambda: _handle_general_question(sender, prompt, phone_id_),
    }

    handler = routes.get(current_step)
    if handler:
        handler()
    else:
        handle_main_menu(sender, prompt, phone_id_)


def _handle_general_question(sender, prompt, phone_id_):
    state = user_states[sender]
    lang = state["language"]
    reply = ask_gemini_general(prompt, lang, sender=sender)
    send(reply, sender, phone_id_)
    _send_more_questions(sender, phone_id_, lang)
    state["step"] = "general_followup"
    save_single_user_state(sender)


# ─────────────────────────────────────────────
#  WhatsApp Cloud API webhook routes
# ─────────────────────────────────────────────

@app.route("/agent-timeout", methods=["POST"])
def agent_timeout():
    try:
        body = request.get_json(force=True) or {}
        user_number = body.get("user_number")
        current_phone_id = body.get("phone_id")
        if not user_number or not current_phone_id:
            return jsonify({"status": "error", "message": "Missing user_number or phone_id"}), 400
        if redis_client:
            raw = redis_client.get(_agent_request_key(user_number))
            if raw:
                request_data = json.loads(raw)
                if request_data.get("status") == "pending":
                    ensure_user_state(user_number)
                    _handle_no_agents_available(user_number, current_phone_id)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logging.error(f"Error in agent_timeout endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("connected.html")
    except Exception:
        return jsonify({
            "service": "Dawa Health — Rudo Assistant",
            "status": "running",
            "content_loaded": CONTENT_LOADED,
            "redis_connected": bool(redis_client),
            "firestore_connected": bool(firestore_db),
            "endpoints": {
                "GET/POST /webhook": "WhatsApp Cloud API webhook",
                "POST /agent-timeout": "Agent handoff timeout callback",
                "POST /api/chat": "Mobile app chat endpoint",
                "POST /api/send-message": "Mobile app send-message endpoint",
                "GET /api/chat-history/<user_id>": "Mobile app chat history",
                "GET /api/health": "Health check",
            },
        })


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        verify_token = os.environ.get("VERIFY_TOKEN", "BOT")
        if mode == "subscribe" and token == verify_token:
            return challenge, 200
        return "Failed", 403

    try:
        data = request.get_json()

        if "entry" in (data or {}):
            for entry in data["entry"]:
                if "changes" not in entry:
                    continue
                for change in entry["changes"]:
                    if "value" not in change:
                        continue
                    value = change["value"]

                    if "messages" in value:
                        for message in value["messages"]:
                            sender = message["from"]
                            phone_id_ = value["metadata"]["phone_number_id"]

                            if message.get("type") == "interactive":
                                interactive = message.get("interactive", {})
                                if interactive.get("type") == "button_reply":
                                    btn_id = interactive["button_reply"]["id"]
                                    if btn_id.startswith("agent_accept:"):
                                        handle_agent_accept(sender, btn_id.split("agent_accept:", 1)[1], phone_id_)
                                    elif btn_id.startswith("agent_reject:"):
                                        handle_agent_reject(sender, btn_id.split("agent_reject:", 1)[1], phone_id_)
                                continue

                            if "text" in message:
                                prompt = message["text"]["body"]

                                agent_phones_norm = {normalize_phone(p) for p in AGENTS.values()}
                                if normalize_phone(sender) in agent_phones_norm:
                                    if relay_agent_message_to_user(sender, prompt, phone_id_):
                                        continue

                                is_new = ensure_user_state(sender)
                                if not is_new:
                                    user_states[sender]["first_message"] = False

                                save_user_conversation(sender, "user", prompt)

                                referral = extract_referral_source(prompt)
                                if referral:
                                    save_referral_source(sender, referral)

                                handle_conversation_state(sender, prompt, phone_id_)
                            else:
                                ensure_user_state(sender)
                                state = user_states.get(sender, {})
                                lang = state.get("language", "english")
                                send("Sorry, I can only process text messages. Please send a text message.", sender, phone_id_)

                    elif "statuses" in value:
                        logging.info("Message status update received, ignoring.")

    except Exception as e:
        logging.error(f"Error in webhook: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
#  Mobile-app JSON API (from the companion build)
#  Reuses the SAME user_states / Redis / Gemini pipeline as WhatsApp,
#  so a user_id can carry state consistently across channels.
# ─────────────────────────────────────────────

def handle_chat_message(user_id: str, message: str, phone_number: str = None) -> str:
    is_new = ensure_user_state(user_id)
    if is_new:
        user_states[user_id]["step"] = "main_menu"
        user_states[user_id]["registered"] = True
        user_states[user_id]["first_message"] = False
        detected = detect_language(message)
        user_states[user_id]["language"] = detected
        if phone_number:
            user_states[user_id]["phone_digits"] = phone_number[-4:]

    save_user_conversation(user_id, "user", message)
    lang = user_states[user_id].get("language", "english")
    reply = ask_gemini_general(message, lang, sender=user_id)
    save_user_conversation(user_id, "bot", reply)
    save_single_user_state(user_id)
    return reply


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_id = data.get("user_id") or data.get("session_id")
        message = data.get("message")
        phone_number = data.get("phone_number")

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        response = handle_chat_message(user_id, message, phone_number)
        return jsonify({"reply": response, "user_id": user_id, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json(force=True)
        user_id = data.get("user_id")
        phone_number = data.get("phone_number")
        message = data.get("message")

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        response = handle_chat_message(user_id, message, phone_number)
        return jsonify({
            "success": True,
            "message": "Message sent successfully",
            "response": response,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logging.error(f"Send message error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    try:
        limit = request.args.get("limit", default=50, type=int)
        redis_history = get_user_conversation(user_id)[-limit:]
        firestore_history = get_chat_history_firestore(user_id, limit) if firestore_db else []
        return jsonify({
            "user_id": user_id,
            "messages": redis_history,
            "firestore_messages": firestore_history,
            "count": len(redis_history),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logging.error(f"Chat history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy",
        "service": "Dawa Health AI Chat API (Rudo)",
        "model": model_name,
        "content_loaded": CONTENT_LOADED,
        "redis": "connected" if redis_client else "not_configured",
        "firestore": "connected" if firestore_db else "not_configured",
        "whatsapp_configured": bool(wa_token and phone_id),
        "gemini_configured": bool(gen_api),
        "timestamp": datetime.now().isoformat(),
    }
    if gen_api:
        try:
            model = genai.GenerativeModel(model_name)
            test_response = model.generate_content("Say OK")
            status["gemini_test"] = (test_response.text or "")[:50]
        except Exception as e:
            status["status"] = "degraded"
            status["gemini_error"] = str(e)
    return jsonify(status), (200 if status["status"] == "healthy" else 503)


# Local dev entry point only — Vercel imports `app` directly (see api/index.py)
if __name__ == "__main__":
    load_user_states()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("FLASK_DEBUG")))
