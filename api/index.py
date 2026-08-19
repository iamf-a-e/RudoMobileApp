"""
Dawa Health — Mobile Chat API
==============================
Single Flask app deployed as a Vercel Python serverless function.
All /api/* traffic is rewritten into this file (see /vercel.json).

This ports the FULL conversation engine from the WhatsApp bot (language
detection w/ LLM fallback, pregnancy-week lookup, cervical cancer Q&A bank,
shop/ordering flow, Gemini general fallback) onto a stateless request/response
HTTP API instead of the WhatsApp Cloud API. WhatsApp-only features (human
agent handoff, interactive buttons, referral-poster tracking, phone-number
registration) are intentionally left out per your instructions.py, which
already says mobile users are pre-authenticated.

State is kept per-user in Redis (Upstash), one key per user — NOT one big
JSON blob — so it scales past a handful of users.
"""

import os
import re
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify
import google.generativeai as genai
from upstash_redis import Redis

from _lib.products_data import products_by_category
from _lib.instructions import (
    instructions as BASE_INSTRUCTIONS,
    DISCLAIMER,
    company_name,
    company_email,
    company_website,
    company_phone,
)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
application = app  # some WSGI adapters look for this name

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

GEN_API = os.environ.get("GEN_API")
if not GEN_API:
    logging.error("GEN_API environment variable not set — Gemini calls will fail")
else:
    genai.configure(api_key=GEN_API)

MODEL_NAME = "gemini-2.5-flash"

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

SUPPORTED_LANGUAGES = ["english", "shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"]

# ─────────────────────────────────────────────
#  Redis (Upstash) — per-user state + optional Firestore chat log
# ─────────────────────────────────────────────

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
else:
    logging.warning("UPSTASH_REDIS_URL / UPSTASH_REDIS_TOKEN not set — state will not persist across requests")

firestore_db = None
try:
    firebase_creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if firebase_creds:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(firebase_creds))
            firebase_admin.initialize_app(cred)
        firestore_db = firestore.client()
        logging.info("Firebase Firestore initialized")
    else:
        logging.info("FIREBASE_SERVICE_ACCOUNT not set — chat history logging disabled")
except Exception as e:
    logging.error(f"Firebase initialization failed: {e}")
    firestore_db = None


def _state_key(user_id):
    return f"mobile_user_state:{user_id}"


def load_user_state(user_id):
    default = {
        "step": "main_menu",
        "language": "english",
        "topic": None,
        "cart": [],
        "first_message": True,
    }
    if not redis_client:
        return default
    try:
        raw = redis_client.get(_state_key(user_id))
        if raw:
            state = json.loads(raw)
            for k, v in default.items():
                state.setdefault(k, v)
            return state
    except Exception as e:
        logging.error(f"Error loading state for {user_id}: {e}")
    return default


def save_user_state(user_id, state):
    if not redis_client:
        return
    try:
        redis_client.set(_state_key(user_id), json.dumps(state), ex=60 * 60 * 24 * 30)
    except Exception as e:
        logging.error(f"Error saving state for {user_id}: {e}")


def get_conversation(user_id):
    if not redis_client:
        return []
    try:
        raw = redis_client.get(f"conversation:{user_id}")
        return json.loads(raw) if raw else []
    except Exception:
        return []


def append_conversation(user_id, role, message):
    if not redis_client:
        return
    try:
        history = get_conversation(user_id)
        history.append({"role": role, "message": str(message), "timestamp": datetime.now().isoformat()})
        history = history[-100:]
        redis_client.set(f"conversation:{user_id}", json.dumps(history), ex=60 * 60 * 24 * 30)
    except Exception as e:
        logging.error(f"Error saving conversation for {user_id}: {e}")


def log_to_firestore(user_id, message, is_user):
    if not firestore_db:
        return
    try:
        from firebase_admin import firestore as fb_firestore

        chat_ref = firestore_db.collection("mobile_chats").document(user_id)
        chat_ref.set(
            {"user_id": user_id, "updated_at": fb_firestore.SERVER_TIMESTAMP, "last_message": message[:200]},
            merge=True,
        )
        chat_ref.collection("messages").add(
            {
                "content": message,
                "is_user": is_user,
                "timestamp": fb_firestore.SERVER_TIMESTAMP,
                "created_at": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logging.error(f"Firestore log failed: {e}")


# ─────────────────────────────────────────────
#  Shared matching helper (ported from WhatsApp bot)
# ─────────────────────────────────────────────

def _contains_signal(prompt_lower, phrases):
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
#  Language detection (keyword scoring + Gemini fallback)
# ─────────────────────────────────────────────

EXACT_MATCHES = {
    "shona": ["mhoro", "mhoroi", "makadini", "hesi", "hapana", "ndizvo", "zvakanaka", "wadini", "kwete"],
    "ndebele": ["sawubona", "salibonani", "unjani", "yebo", "ngiyabonga", "hatshi"],
    "bemba": ["mwaiseni", "ulishani", "shani", "mukwai", "sana"],
    "chinyanja": ["moni", "zikomo", "pepani", "chonde", "eyaa"],
    "tonga": ["mwabuka", "mwalandwa", "ndatotela", "kapati", "mbuti"],
    "lozi": ["ndalumba", "haa", "kacenu", "muzuhile"],
}

LANGUAGE_KEYWORDS = {
    "shona": ["mhoro", "mhoroi", "makadini", "zvakanaka", "ndatenda", "pamuviri", "chibereko",
              "ndinoda", "kwete", "hapana", "ndine", "ndiri", "sei", "vanhu", "muviri"],
    "ndebele": ["sawubona", "salibonani", "unjani", "ngiyabonga", "isisu", "umntwana", "umhlaza",
                "angikwazi", "ngifuna", "abantu", "impela"],
    "chinyanja": ["moni", "zikomo", "pepani", "ndapota", "matenda", "kansa", "dokotala", "magazi",
                  "ndikufuna", "mimba", "bwanji", "ndimva"],
    "lozi": ["ndalumba", "kuhula", "maviki", "mutango", "kacenu", "musimbi", "bulwazi", "cwale"],
    "bemba": ["mwaiseni", "natotela", "twatotela", "mukwai", "ubushiku", "icisungu", "shani"],
    "tonga": ["ndalumba", "mubuzyo", "kapati", "mutumbu", "dokota", "buti", "makani", "buumi"],
    "english": ["what", "how", "when", "why", "where", "signs", "symptoms", "please", "thank",
                "help", "during"],
}

LANGUAGE_PHRASES = {
    "chinyanja": ["muli bwanji", "uli ndi chani", "ndili bwino"],
    "shona": ["makadini", "zvakanaka sei", "ndiriku"],
    "ndebele": ["unjani wena", "sicela ungichazele"],
    "lozi": ["uli bwanji", "ha ndi zibi"],
    "bemba": ["muli shani", "napapata"],
    "tonga": ["mmuli buti", "ndakomba"],
    "english": ["how are you", "what is", "can you", "tell me", "i need", "i want"],
}


def _llm_detect_language(message):
    try:
        classifier_prompt = (
            "Identify which ONE of these languages the following message is written in: "
            "english, shona, ndebele, chinyanja, bemba, tonga, lozi. These are languages "
            "spoken in Zimbabwe and Zambia. Reply with ONLY the single lowercase language "
            f"name, nothing else.\n\nMessage: \"{message}\""
        )
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={"temperature": 0, "max_output_tokens": 10},
            safety_settings=safety_settings,
        )
        response = model.generate_content(classifier_prompt)
        guess = re.sub(r"[^a-z]", "", response.text.strip().lower())
        return guess if guess in SUPPORTED_LANGUAGES else None
    except Exception as e:
        logging.error(f"[_llm_detect_language] {e}")
        return None


def detect_language(message, current_lang="english"):
    message_lower = message.lower().strip()
    if not message_lower or message_lower.isdigit():
        return current_lang

    for lang, words in EXACT_MATCHES.items():
        if message_lower in words:
            return lang

    scores = {lang: 0 for lang in LANGUAGE_KEYWORDS}
    for lang, phrases in LANGUAGE_PHRASES.items():
        for phrase in phrases:
            if phrase in message_lower:
                scores[lang] = scores.get(lang, 0) + 5
    for lang, keywords in LANGUAGE_KEYWORDS.items():
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
        guess = _llm_detect_language(message)
        if guess:
            return guess

    if all(ord(c) < 128 for c in message_lower):
        return current_lang if current_lang != "english" else "english"

    return "english"


# ─────────────────────────────────────────────
#  Per-language data lookup (falls back to English if the language
#  module doesn't exist yet — drop pregnancy_data_shona.py etc. in
#  this folder, matching the WhatsApp bot's file naming, to enable it)
# ─────────────────────────────────────────────

def get_pregnancy_data(language):
    if language and language != "english":
        try:
            mod = __import__(f"_lib.pregnancy_data_{language}", fromlist=[f"pregnancy_data_{language}"])
            return getattr(mod, f"pregnancy_data_{language}")
        except Exception:
            logging.info(f"No _lib/pregnancy_data_{language}.py found, falling back to English")
    from _lib.pregnancy_data import pregnancy_data
    return pregnancy_data


def get_cervical_data(language):
    if language and language != "english":
        try:
            mod = __import__(f"_lib.cervical_cancer_data_{language}", fromlist=[f"cervical_cancer_data_{language}"])
            return getattr(mod, f"cervical_cancer_data_{language}")
        except Exception:
            logging.info(f"No _lib/cervical_cancer_data_{language}.py found, falling back to English")
    from _lib.cervical_cancer_data import cervical_cancer_data
    return cervical_cancer_data


WEEK_MARKER = {
    "shona": "Vhiki", "ndebele": "Iviki", "chinyanja": "Sabata", "lozi": "Sunda",
    "bemba": "Umulungu", "tonga": "Nhwiiiki", "english": "Week",
}


# ─────────────────────────────────────────────
#  Gemini prompt builders
# ─────────────────────────────────────────────

LANG_ENFORCE = {
    "shona": "Pindura muchiShona chete.",
    "ndebele": "Phendula ngesiNdebele kuphela.",
    "chinyanja": "Yankhani mu Chinyanja basi.",
    "lozi": "Arabela ka Silozi feela.",
    "bemba": "Yasuka mu Cibemba fye.",
    "tonga": "Mupandule mu Chitonga buyo.",
    "english": "Respond in English only.",
}

FALLBACK_MSG = {
    "shona": "Pane dambudziko pakupindura mubvunzo wako.",
    "ndebele": "Kunenkinga ekuphenduleni umbuzo wakho.",
    "chinyanja": "Pali vuto popanga yankho la funso lanu.",
    "tonga": "Kuli ipenzi mukupandula mubuzyo wanu.",
    "bemba": "Cabulanda, kuliko ubwafya pakwasuka ilipusho lyobe.",
    "lozi": "Ku na bothata ka ku arabela lipuzo la hao.",
    "english": "Sorry, there was a problem getting an answer.",
}


def build_context(user_id, max_turns=6):
    history = get_conversation(user_id)
    if not history:
        return ""
    recent = history[-(max_turns * 2 + 1):-1]
    lines = []
    for entry in recent:
        msg = entry.get("message", "").strip()
        if not msg or (entry.get("role") == "bot" and len(msg) < 20):
            continue
        tag = "User" if entry.get("role") == "user" else "Assistant"
        lines.append(f"{tag}: {msg}")
    return ("Previous conversation:\n" + "\n".join(lines) + "\n\n") if lines else ""


def ask_gemini(question, lang, user_id, topic_hint=""):
    lang_enforce = LANG_ENFORCE.get(lang, LANG_ENFORCE["english"])
    fallback = FALLBACK_MSG.get(lang, FALLBACK_MSG["english"])
    context = build_context(user_id)

    prompt = (
        f"{BASE_INSTRUCTIONS}\n\n"
        f"{topic_hint}\n\n"
        f"{context}"
        f"Current question: {question}\n\n"
        f"{lang_enforce} Do not start with filler words like 'Okay' or 'Sure' — answer directly. "
        f"Keep it concise (a short paragraph or a few bullet points)."
    )

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config, safety_settings=safety_settings)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip() + DISCLAIMER
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini] {type(e).__name__}: {e}")
        return fallback


# ─────────────────────────────────────────────
#  Conversation engine
#  Each handler appends to `out` (list of strings) instead of calling a
#  WhatsApp send() — the API layer joins them into one reply.
# ─────────────────────────────────────────────

GREETING_WORDS = [
    "hi", "hello", "hey", "hie", "good morning", "good afternoon", "good evening",
    "mhoro", "mhoroi", "hesi", "makadini", "sawubona", "salibonani",
    "moni", "muli bwanji", "mwabuka", "mwaiseni", "muli shani", "mwa bona",
]
RESET_KEYWORDS = ["start over", "restart", "new conversation", "main menu", "menu", "reset"]
YES_WORDS = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde", "yebo"]
NO_WORDS = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "cha", "ayi", "not really"]

MATERNAL_KEYWORDS = ["pregnan", "pamuviri", "pakati", "baby", "maternal", "nhumbu", "trimester", "prenatal"]
CERVICAL_KEYWORDS = ["cervical", "hpv", "cervix", "gomarara", "umhlaza", "kansa", "kankere"]
SHOP_KEYWORDS = ["buy", "purchase", "order", "shop", "product", "price", "cost", "zvigadzirwa", "gula", "landa", "shita"]


def greet_text(lang):
    return {
        "shona": "Mhoroi! Ndini Rudo, mubatsiri weDawa Health. Ndingakubatsirei nhasi?",
        "ndebele": "Sawubona! Ngingu Rudo, umsizi we-Dawa Health. Ngingakusiza ngani namuhla?",
        "chinyanja": "Moni! Ndine Rudo, wothandiza wa Dawa Health. Ndingakuthandizireni lero?",
        "lozi": "Mwa bona! Mina ki Rudo, mubasi wa Dawa Health. Nka ku thusa ka mini sunu?",
        "tonga": "Mwabuka buti! Ndime Rudo, wakugwasya wa Dawa Health. Nga ndamugwasya buti sunu?",
        "bemba": "Muli shani! Nine Rudo, wakufwailisha wa Dawa Health. Bushe kuti namwafwa shani lelo?",
    }.get(lang, "Hello! I'm Rudo, Dawa Health's virtual assistant. How can I help you today?")


def ask_week_text(lang):
    return {
        "shona": "Ndapota isa vhiki rako repamuviri (1-40).",
        "ndebele": "Sicela ufake iviki lakho lokukhulelwa (1-40).",
        "chinyanja": "Chonde lowetsani sabata lanu la pakati (1-40).",
        "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana (1-40).",
        "tonga": "Amubike namba yamaviki aanu (1-40).",
        "bemba": "Ingishenimo umulungu wenu uwa pabukulu (1-40).",
    }.get(lang, "Please enter your pregnancy week number (1-40).")


def week_not_found_text(lang, week):
    return {
        "shona": f"Hapana ruzivo rwevhiki {week}.",
        "ndebele": f"Alukho ulwazi lwe-iviki {week}.",
        "chinyanja": f"Palibe zambiri za sabata {week}.",
        "lozi": f"Ha ku na taba za sunda {week}.",
        "tonga": f"Kunyina zinji zya wiiki {week}.",
        "bemba": f"Tapali ifilifyonse pa mulungu {week}.",
    }.get(lang, f"No data available for week {week}.")


def another_week_text(lang):
    return {
        "shona": "Ungada here kudzidza nezve rimwe vhiki repamuviri?",
        "ndebele": "Ungathanda ukufunda ngelinye iviki lokukhulelwa?",
        "chinyanja": "Kodi mukufuna kudziwa za sabata lina la pakati?",
        "lozi": "Kana u bata ku ithuta ka za sunda ye ñwi ya buimana?",
        "tonga": "Mulakonzya kuyanda kuzyiba zya wiiki imwi ya bubulemi?",
        "bemba": "Kuti mwafwaya ukwishiba pa mulungu umbi uwa pabukulu?",
    }.get(lang, "Would you like to learn about another pregnancy week?")


def keep_learning_text(lang):
    return {
        "shona": "Ungada here kuramba uchidzidza zvimwe zvecervical cancer?",
        "ndebele": "Ungathanda ukuqhubeka nokufunda okunye nge-cervical cancer?",
        "chinyanja": "Kodi mukufuna kupitiriza kuphunzira zambiri za cervical cancer?",
        "lozi": "Kana u bata ku zwelapili ku ithuta ze ñwi ka za kankere ya sibeleko?",
        "tonga": "Mulakonzya kuyanda kwiya zimwi zya kansa ya mulomo wa cibeleko?",
        "bemba": "Bushe ulefwaya ukukonkanyapo ukusambililapo ifingi pali cervical cancer?",
    }.get(lang, "Would you like to keep learning about cervical cancer?")


def shop_categories_text(lang):
    header = {
        "shona": "🛒 Makategori eZvigadzirwa:\n",
        "ndebele": "🛒 Imigqa Yemikhiqizo:\n",
        "chinyanja": "🛒 Mitundu ya Zinthu:\n",
        "lozi": "🛒 Mibeko ya Swakupila:\n",
        "tonga": "🛒 Misela ya Zintu:\n",
        "bemba": "🛒 Imisango ya Fipe:\n",
    }.get(lang, "🛒 Product Categories:\n")

    lines = [header]
    for idx, (cat_name, items) in enumerate(products_by_category.items(), 1):
        lines.append(f"*{idx}. {cat_name}*")
        for item in items:
            lines.append(f"   • {item['name']} — {item['price']} ({item['availability']})")
        lines.append("")

    prompt = {
        "shona": "Tumira nhamba yekategori kana zita rechigadzirwa chaunoda kuodha.",
        "ndebele": "Thumela inombolo yomugqa noma igama lomkhiqizo ofuna ukuwodha.",
        "chinyanja": "Tumizani nambala ya gulu kapena dzina la chinthu mukufuna kugula.",
        "lozi": "Lumeza nomolo ya sibaka kamba libizo la swakupila u bata ku landa.",
        "tonga": "Tumizya nambala ya musela naa zina lya cintu ncimuyanda kuula.",
        "bemba": "Tuma nambala ya fiputulwa nangu ishina lya fintu mulefwaya ukushita.",
    }.get(lang, "Reply with a category number, or the name of the product you'd like to order.")
    lines.append(prompt)
    return "\n".join(lines)


def find_product(prompt_lower):
    all_products = [p for items in products_by_category.values() for p in items]
    return next(
        (p for p in all_products if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()),
        None,
    )


def handle_turn(user_id, message, state, out):
    """Core router. Mutates `state` and appends reply strings to `out`."""
    lang = state["language"]
    prompt_lower = message.lower().strip()
    step = state.get("step", "main_menu")

    is_greeting = _contains_signal(prompt_lower, GREETING_WORDS)
    is_reset = _contains_signal(prompt_lower, RESET_KEYWORDS)
    if is_greeting or is_reset:
        state.update({"step": "main_menu", "topic": None, "cart": state.get("cart", [])})
        out.append(greet_text(lang))
        return

    # ---- pregnancy week flow ----
    if step == "ask_week":
        digits = re.sub(r"\D", "", prompt_lower)
        if digits and 1 <= int(digits) <= 40:
            week = int(digits)
            marker = WEEK_MARKER.get(lang, "Week")
            info_text = get_pregnancy_data(lang)
            pattern = rf"\*{marker} {week}:.*?(?=\*{marker} {week + 1}:|\Z)"
            match = re.search(pattern, info_text, re.S)
            if match:
                out.append(match.group(0).strip())
            else:
                out.append(week_not_found_text(lang, week))
            out.append(another_week_text(lang))
            state["step"] = "ask_another_week"
        else:
            out.append(ask_week_text(lang))
        return

    if step == "ask_another_week":
        if _contains_signal(prompt_lower, YES_WORDS):
            state["step"] = "ask_week"
            out.append(ask_week_text(lang))
        elif _contains_signal(prompt_lower, NO_WORDS):
            state["step"] = "main_menu"
            state["topic"] = None
            out.append(ask_gemini(
                "The user is done learning about pregnancy weeks for now — ask if there's "
                "anything else you can help with today.",
                lang, user_id,
            ))
        else:
            out.append(another_week_text(lang))
        return

    # ---- cervical cancer flow ----
    if step == "cervical_question_number":
        digits = re.sub(r"\D", "", prompt_lower)
        if digits and 1 <= int(digits) <= 100:
            qnum = int(digits)
            data = get_cervical_data(lang)
            found = False
            for i, item in enumerate(data):
                if f"*Question {qnum}:" in str(item):
                    content = str(item)
                    if i + 1 < len(data) and "Answer" in str(data[i + 1]):
                        content += "\n" + str(data[i + 1])
                    out.append(content)
                    found = True
                    break
            if not found:
                out.append(f"Sorry, I couldn't find question number {qnum}. Try 1-{len(data) // 2 or 1}.")
            out.append(keep_learning_text(lang))
            state["step"] = "keep_learning"
        else:
            out.append("Please enter a question number between 1 and 100.")
        return

    if step == "keep_learning":
        if _contains_signal(prompt_lower, YES_WORDS):
            state["step"] = "cervical_question_number"
            out.append("Enter a question number from 1 to 100:")
        elif _contains_signal(prompt_lower, NO_WORDS):
            state["step"] = "main_menu"
            state["topic"] = None
            out.append(ask_gemini(
                "The user is done learning about cervical cancer for now — ask if there's "
                "anything else you can help with today.",
                lang, user_id,
            ))
        else:
            out.append(keep_learning_text(lang))
        return

    # ---- shop flow ----
    if step == "shop_browse":
        if prompt_lower.isdigit():
            categories = list(products_by_category.keys())
            idx = int(prompt_lower) - 1
            if 0 <= idx < len(categories):
                cat = categories[idx]
                items = products_by_category[cat]
                lines = [f"🏥 *{cat}*\n"]
                for i, item in enumerate(items, 1):
                    lines.append(f"{i}. {item['name']} — {item['price']} ({item['availability']})\n   {item['description']}")
                lines.append("\nTell us the product name to order it, or 'no' to go back.")
                out.append("\n".join(lines))
                state["step"] = "shop_product_name"
                return
            out.append(f"Please choose a category between 1 and {len(categories)}.")
            return
        matched = find_product(prompt_lower)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            state["step"] = "shop_quantity"
            out.append(f"Great choice! How many *{matched['name']}* would you like?")
            return
        out.append(shop_categories_text(lang))
        return

    if step == "shop_product_name":
        matched = find_product(prompt_lower)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            state["step"] = "shop_quantity"
            out.append(f"Great choice! How many *{matched['name']}* would you like?")
        elif _contains_signal(prompt_lower, NO_WORDS):
            state["step"] = "shop_browse"
            out.append(shop_categories_text(lang))
        else:
            out.append("I couldn't find that product. Please type the exact product name from the list.")
        return

    if step == "shop_quantity":
        qty_match = re.search(r"\d+", prompt_lower)
        if qty_match:
            qty = int(qty_match.group())
            cart = state.setdefault("cart", [])
            cart.append({
                "product": state.get("shop_selected_product", "Unknown"),
                "price": state.get("shop_selected_price", "N/A"),
                "quantity": qty,
            })
            out.append(f"✅ *{state.get('shop_selected_product')}* x{qty} added to your order! "
                        "Would you like to add anything else? (yes/no)")
            state["step"] = "shop_add_more"
        else:
            out.append("Please enter a number (e.g. 1, 2, 3).")
        return

    if step == "shop_add_more":
        matched = find_product(prompt_lower)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            state["step"] = "shop_quantity"
            out.append(f"How many *{matched['name']}* would you like?")
            return
        if _contains_signal(prompt_lower, YES_WORDS):
            state["step"] = "shop_browse"
            out.append(shop_categories_text(lang))
            return
        if _contains_signal(prompt_lower, NO_WORDS):
            state["step"] = "shop_address"
            out.append("Great! Please provide your delivery address (town, area, and any helpful details).")
            return
        out.append("Please reply 'yes' to add more, or 'no' to check out.")
        return

    if step == "shop_address":
        address = message.strip()
        cart = state.get("cart", [])
        if redis_client:
            try:
                for item in cart:
                    order = {
                        "user_id": user_id, "product": item["product"], "price": item["price"],
                        "quantity": item["quantity"], "address": address,
                        "timestamp": datetime.now().isoformat(), "status": "pending",
                    }
                    key = f"orders:{user_id}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    redis_client.set(key, json.dumps(order))
            except Exception as e:
                logging.error(f"Error saving order: {e}")
        lines = ["✅ *Order Confirmed!*\n"]
        for item in cart:
            lines.append(f"  📦 {item['product']} x{item['quantity']} — {item['price']}")
        lines.append(f"\n📍 Delivery address: {address}")
        lines.append("We'll be in touch shortly. Thank you! 😊")
        out.append("\n".join(lines))
        state.update({"step": "main_menu", "topic": None, "cart": []})
        return

    # ---- main menu / topic routing ----
    if _contains_signal(prompt_lower, SHOP_KEYWORDS):
        state["step"] = "shop_browse"
        out.append(shop_categories_text(lang))
        return

    if _contains_signal(prompt_lower, MATERNAL_KEYWORDS):
        state["topic"] = "maternal"
        state["step"] = "ask_week"
        out.append(ask_week_text(lang))
        return

    if _contains_signal(prompt_lower, CERVICAL_KEYWORDS):
        state["topic"] = "cervical"
        out.append(ask_gemini(message, lang, user_id, topic_hint="Topic: cervical cancer."))
        out.append(keep_learning_text(lang))
        state["step"] = "keep_learning"
        return

    # general fallback via Gemini
    out.append(ask_gemini(message, lang, user_id))
    state["step"] = "main_menu"


# ─────────────────────────────────────────────
#  API routes
# ─────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get("user_id") or data.get("session_id")
        message = data.get("message")
        forced_lang = data.get("language")

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        state = load_user_state(user_id)

        if forced_lang in SUPPORTED_LANGUAGES:
            state["language"] = forced_lang
        elif not state.get("first_message"):
            state["language"] = detect_language(message, state.get("language", "english"))
        else:
            state["language"] = detect_language(message, "english")
        state["first_message"] = False

        append_conversation(user_id, "user", message)
        log_to_firestore(user_id, message, is_user=True)

        out = []
        handle_turn(user_id, message, state, out)
        reply = "\n\n".join(m for m in out if m)

        append_conversation(user_id, "bot", reply)
        log_to_firestore(user_id, reply, is_user=False)
        save_user_state(user_id, state)

        return jsonify({
            "reply": reply,
            "messages": out,
            "user_id": user_id,
            "language": state["language"],
            "step": state["step"],
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logging.error(f"/api/chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Kept for parity with the older mobile client that posts here instead
@app.route("/api/send-message", methods=["POST"])
def send_message():
    return chat()


@app.route("/api/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    try:
        limit = request.args.get("limit", default=50, type=int)
        history = get_conversation(user_id)[-limit:]
        return jsonify({"user_id": user_id, "messages": history, "count": len(history)})
    except Exception as e:
        logging.error(f"/api/chat-history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset/<user_id>", methods=["POST"])
def reset_user(user_id):
    state = {"step": "main_menu", "language": "english", "topic": None, "cart": [], "first_message": True}
    save_user_state(user_id, state)
    return jsonify({"status": "ok", "user_id": user_id})


@app.route("/api/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy",
        "service": "Dawa Health Mobile Chat API",
        "model": MODEL_NAME,
        "redis": "connected" if redis_client else "not_configured",
        "firestore": "connected" if firestore_db else "not_configured",
        "gemini_key_present": bool(GEN_API),
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(status)


@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
def home():
    return jsonify({
        "service": "Dawa Health Mobile Chat API",
        "version": "2.0.0",
        "endpoints": {
            "POST /api/chat": "Main chat endpoint",
            "POST /api/send-message": "Alias of /api/chat",
            "GET /api/chat-history/<user_id>": "Get chat history",
            "POST /api/reset/<user_id>": "Reset a user's conversation state",
            "GET /api/health": "Health check",
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
