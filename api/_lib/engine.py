"""
Dawa Health — Mobile Chat Engine (shared logic)
=================================================
This module holds ALL the conversation logic (language detection, pregnancy
week Q&A, cervical cancer Q&A, shop/ordering, Gemini calls, Redis state).

It has NO Flask app / routes of its own — each file directly under /api/
(chat.py, health.py, etc.) is a thin wrapper that imports this module, so
every route is its own physical file matching Vercel's native zero-config
Python routing. This avoids relying on vercel.json rewrites, which behave
unpredictably in mixed Next.js + Python projects.


"""

import os
import sys

# This file lives at api/_lib/engine.py. Vercel imports api/*.py entrypoints
# via importlib without adding directories to sys.path, so make sure the
# `api/` directory (this file's grandparent) is on sys.path before doing any
# `from _lib...` import below — otherwise `_lib` won't resolve as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import sys
import json
import logging
from datetime import datetime

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
from _lib.sahara_client import (
    transcribe_audio,
    VOICE_SUPPORTED_LANGUAGES,
    get_voice_unsupported_response,
)

logging.basicConfig(level=logging.INFO)

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
        "current_week": None,
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

    # Common English filler words (how/please/help/...) show up inside
    # otherwise-local-language messages too, and used to tie with the real
    # signal — which then defaulted to staying "english". Any clear,
    # UNIQUE non-English top scorer should win over an English tie.
    non_english_scores = {l: s for l, s in scores.items() if l != "english" and s > 0}
    if non_english_scores:
        top_score = max(non_english_scores.values())
        candidates = [l for l, s in non_english_scores.items() if s == top_score]
        if len(candidates) == 1 and top_score >= scores.get("english", 0):
            return candidates[0]

    # Weak or ambiguous local signal (including none at all) — ask the
    # Gemini classifier rather than silently defaulting to English.
    words_in_msg = re.findall(r"[a-z]+", message_lower)
    if len(words_in_msg) >= 2:
        guess = _llm_detect_language(message)
        if guess:
            return guess

    max_score = max(scores.values()) if scores else 0
    if max_score > 0:
        top_langs = [lang for lang, s in scores.items() if s == max_score]
        if current_lang in top_langs:
            return current_lang
        if len(top_langs) == 1:
            return top_langs[0]
        return current_lang

    if all(ord(c) < 128 for c in message_lower):
        return current_lang

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
    "shona": "Pindura muchiShona chete. (Respond ONLY in Shona — do not use English.)",
    "ndebele": "Phendula ngesiNdebele kuphela. (Respond ONLY in Ndebele — do not use English.)",
    "chinyanja": "Yankhani mu Chinyanja basi. (Respond ONLY in Chinyanja/Nyanja — do not use English.)",
    "lozi": "Arabela ka Silozi feela. (Respond ONLY in Lozi — do not use English.)",
    "bemba": "Yasuka mu Cibemba fye. (Respond ONLY in Bemba — do not use English.)",
    "tonga": "Mupandule mu Chitonga buyo. (Respond ONLY in Tonga — do not use English.)",
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
        f"IMPORTANT LANGUAGE INSTRUCTION: {lang_enforce}\n\n"
        f"{BASE_INSTRUCTIONS}\n\n"
        f"{topic_hint}\n\n"
        f"{context}"
        f"Current question: {question}\n\n"
        f"REMINDER — {lang_enforce} "
        f"Do not start with filler words like 'Okay' or 'Sure' — answer directly. "
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
#  Grounding helpers — pull the relevant chunk(s) out of the existing
#  pregnancy_data_*/cervical_cancer_data_* files so Gemini answers from
#  our own content instead of its open-ended knowledge.
# ─────────────────────────────────────────────

def extract_week_number(message):
    """Pull a plausible pregnancy-week number (1-40) out of free text."""
    for m in re.finditer(r"\b(\d{1,2})\b", message):
        n = int(m.group(1))
        if 1 <= n <= 40:
            return n
    return None


def _parse_week_entries(info_text, marker):
    """Split the raw pregnancy_data text into (week_num, content) pairs."""
    pattern = rf"\*{re.escape(marker)} (\d+):(.*?)(?=\*{re.escape(marker)} \d+:|\Z)"
    return [(int(num), content.strip()) for num, content in re.findall(pattern, info_text, re.S)]


def build_pregnancy_grounding(message, lang, state):
    """
    Return (grounding_text, matched_week) for the pregnancy topic.
    Prefers an explicit week number in this message, then falls back to the
    week last discussed, then a light keyword match across all weeks.
    """
    marker = WEEK_MARKER.get(lang, "Week")
    info_text = get_pregnancy_data(lang)
    entries = _parse_week_entries(info_text, marker)
    if not entries:
        return "", None

    week = extract_week_number(message)
    if week is None:
        # No new week mentioned — if the user is following up ("what about
        # symptoms?") stick with the week they were just on.
        week = state.get("current_week")

    if week:
        match = next(((n, c) for n, c in entries if n == week), None)
        if match:
            state["current_week"] = match[0]
            return f"*{marker} {match[0]}:{match[1]}", match[0]

    # No usable week — light keyword overlap against all week entries so we
    # can still hand Gemini something relevant if the question names a
    # symptom, food, milestone, etc.
    words = set(re.findall(r"[a-z]+", message.lower()))
    scored = []
    for n, c in entries:
        overlap = len(words & set(re.findall(r"[a-z]+", c.lower())))
        if overlap:
            scored.append((overlap, n, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    if top:
        text = "\n\n".join(f"*{marker} {n}:{c}" for _, n, c in top)
        return text, None
    return "", None


def _parse_qa_entries(data):
    """
    cervical_cancer_data is a flat list where a '*Question N: ...*' item is
    normally followed by its answer item — pair them up.
    """
    pairs = []
    i = 0
    while i < len(data):
        item = str(data[i])
        m = re.search(r"\*Question (\d+):", item)
        if m:
            qnum = int(m.group(1))
            answer = ""
            if i + 1 < len(data) and "Answer" in str(data[i + 1]):
                answer = str(data[i + 1])
                i += 1
            pairs.append((qnum, item, answer))
        i += 1
    return pairs


def build_cervical_grounding(message):
    """Return grounding text for the cervical-cancer topic: an explicit
    question number if named, otherwise the best keyword-matched Q&A pairs."""
    data = get_cervical_data("english")  # source data is keyed by question number regardless of language
    pairs = _parse_qa_entries(data)
    if not pairs:
        return ""

    m = re.search(r"\b(\d{1,3})\b", message)
    if m:
        qnum = int(m.group(1))
        match = next((p for p in pairs if p[0] == qnum), None)
        if match:
            return f"{match[1]}\n{match[2]}"

    words = set(re.findall(r"[a-z]+", message.lower()))
    scored = []
    for qnum, q, a in pairs:
        overlap = len(words & set(re.findall(r"[a-z]+", (q + " " + a).lower())))
        if overlap:
            scored.append((overlap, q, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    return "\n\n".join(f"{q}\n{a}" for _, q, a in top)


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
        state.update({"step": "main_menu", "topic": None, "current_week": None, "cart": state.get("cart", [])})
        out.append(greet_text(lang))
        return

    # ---- shop flow (scripted — exact prices/cart, no Gemini) ----
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
        state.update({"step": "main_menu", "topic": None, "current_week": None, "cart": []})
        return

    # ---- shop entry point ----
    if _contains_signal(prompt_lower, SHOP_KEYWORDS):
        state["step"] = "shop_browse"
        out.append(shop_categories_text(lang))
        return

    # ---- pregnancy / maternal topic — Gemini, grounded in pregnancy_data ----
    if _contains_signal(prompt_lower, MATERNAL_KEYWORDS) or state.get("topic") == "maternal":
        state["topic"] = "maternal"
        state["step"] = "topic_chat"
        grounding, week = build_pregnancy_grounding(message, lang, state)
        if grounding:
            hint = (
                "Topic: pregnancy / maternal health. Base your answer ONLY on the reference "
                "information below — do not invent facts beyond it. If the user hasn't named "
                "a pregnancy week and the reference doesn't cover their question, say so briefly "
                "and answer generally and safely. After answering, you may naturally invite them "
                "to ask about another week or topic.\n\nReference:\n" + grounding
            )
        else:
            hint = (
                "Topic: pregnancy / maternal health. The user hasn't specified a pregnancy week "
                "and nothing in the reference data matched their question — ask which week of "
                "pregnancy they are in so you can give week-specific guidance, or answer generally "
                "and safely if the question isn't week-specific."
            )
        out.append(ask_gemini(message, lang, user_id, topic_hint=hint))
        return

    # ---- cervical cancer topic — Gemini, grounded in cervical_cancer_data ----
    if _contains_signal(prompt_lower, CERVICAL_KEYWORDS) or state.get("topic") == "cervical":
        state["topic"] = "cervical"
        state["step"] = "topic_chat"
        grounding = build_cervical_grounding(message)
        if grounding:
            hint = (
                "Topic: cervical cancer. Base your answer ONLY on the reference Q&A below — do "
                "not invent medical facts beyond it. After answering, you may naturally invite "
                "the user to ask another cervical-cancer question.\n\nReference:\n" + grounding
            )
        else:
            hint = (
                "Topic: cervical cancer. Nothing in the reference data closely matched this "
                "question — answer helpfully and safely, keep it general, and suggest they see "
                "a healthcare provider for anything specific to their situation."
            )
        out.append(ask_gemini(message, lang, user_id, topic_hint=hint))
        return

    # ---- general fallback via Gemini ----
    state["topic"] = None
    state["step"] = "main_menu"
    out.append(ask_gemini(message, lang, user_id))


# ─────────────────────────────────────────────
#  Public functions — called by the thin api/*.py route files
# ─────────────────────────────────────────────

def process_chat(user_id, message, forced_lang=None):
    """Runs one turn of conversation for user_id. Returns a JSON-serializable dict."""
    state = load_user_state(user_id)

    if forced_lang in SUPPORTED_LANGUAGES:
        state["language"] = forced_lang
    else:
        state["language"] = detect_language(message, state.get("language", "english"))
    state["first_message"] = False

    append_conversation(user_id, "user", message)
    log_to_firestore(user_id, message, is_user=True)

    out = []
    handle_turn(user_id, message, state, out)
    reply = "\n\n".join(m for m in out if m)

    append_conversation(user_id, "bot", reply)
    log_to_firestore(user_id, reply, is_user=False)
    save_user_state(user_id, state)

    return {
        "reply": reply,
        "messages": out,
        "user_id": user_id,
        "language": state["language"],
        "step": state["step"],
        "timestamp": datetime.now().isoformat(),
    } 


def process_voice_chat(user_id, audio_bytes, filename="voice_note", mime_type="audio/wav"):
    """
    Voice equivalent of process_chat: audio in, text reply out.

    Sahara's TTS voice list only covers english/shona out of our seven
    supported languages — see VOICE_SUPPORTED_LANGUAGES in sahara_client.
    A user whose known language isn't voice-supported gets an English
    apology + redirect to text, without ever calling Sahara. A brand-new
    user (state defaults to "english") will pass the gate on their very
    first voice message even if they actually speak an unsupported
    language — that first transcription may come back poor/garbled; once
    they've used text once, state["language"] is set correctly for future
    voice notes.
    """
    state = load_user_state(user_id)
    known_lang = state.get("language", "english")

    if known_lang not in VOICE_SUPPORTED_LANGUAGES:
        return get_voice_unsupported_response(known_lang)

    transcript, file_id = transcribe_audio(audio_bytes, filename, mime_type, language_hint=known_lang)

    if not transcript:
        error_text = FALLBACK_MSG.get(known_lang, FALLBACK_MSG["english"])
        return {"reply": error_text, "user_id": user_id, "error": "transcription_failed"}

    # Reuse the full existing grounded text pipeline unchanged
    result = process_chat(user_id, transcript, forced_lang=None)
    result["transcript"] = transcript
    result["sahara_file_id"] = file_id
    return result


def get_history(user_id, limit=50):
    return get_conversation(user_id)[-limit:]


def reset_state(user_id):
    state = {"step": "main_menu", "language": "english", "topic": None, "current_week": None, "cart": [], "first_message": True}
    save_user_state(user_id, state)
    return {"status": "ok", "user_id": user_id}


def health_status():
    return {
        "status": "healthy",
        "service": "Dawa Health Mobile Chat API",
        "model": MODEL_NAME,
        "redis": "connected" if redis_client else "not_configured",
        "firestore": "connected" if firestore_db else "not_configured",
        "gemini_key_present": bool(GEN_API),
        "sahara_key_present": bool(os.environ.get("SAHARA_API_KEY")),
        "timestamp": datetime.now().isoformat(),
    }
